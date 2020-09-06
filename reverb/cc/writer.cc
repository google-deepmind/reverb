// Copyright 2019 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "reverb/cc/writer.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/bind_front.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/tensor_compression.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace deepmind {
namespace reverb {
namespace {

int PositiveModulo(int value, int divisor) {
  if (divisor == 0) return value;

  if ((value > 0) == (divisor > 0)) {
    // value and divisor have the same sign.
    return value % divisor;
  } else {
    // value and divisor have different signs.
    return (value % divisor + divisor) % divisor;
  }
}

}  // namespace

Writer::Writer(std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
               int chunk_length, int max_timesteps, bool delta_encoded,
               std::shared_ptr<internal::FlatSignatureMap> signatures,
               absl::optional<int> max_in_flight_items)
    : stub_(std::move(stub)),
      chunk_length_(chunk_length),
      max_timesteps_(max_timesteps),
      delta_encoded_(delta_encoded),
      max_in_flight_items_(std::move(max_in_flight_items)),
      num_items_in_flight_(0),
      signatures_(std::move(signatures)),
      next_chunk_key_(NewID()),
      episode_id_(NewID()),
      index_within_episode_(0),
      closed_(false),
      inserted_dtypes_and_shapes_(max_timesteps) {}

Writer::~Writer() {
  if (!closed_) Close().IgnoreError();
}

tensorflow::Status Writer::Append(std::vector<tensorflow::Tensor> data) {
  if (closed_) {
    return tensorflow::errors::FailedPrecondition(
        "Calling method Append after Close has been called");
  }
  if (!buffer_.empty() && buffer_.front().size() != data.size()) {
    return tensorflow::errors::InvalidArgument(
        "Number of tensors per timestep was inconsistent. Previously it was ",
        buffer_.front().size(), ", but is now ", data.size(), ".");
  }

  // Store flattened signature into inserted_dtypes_and_shapes_
  internal::DtypesAndShapes dtypes_and_shapes_t(0);
  dtypes_and_shapes_t->reserve(data.size());
  for (const auto& t : data) {
    dtypes_and_shapes_t->push_back(
        {/*name=*/"", t.dtype(), tensorflow::PartialTensorShape(t.shape())});
  }
  std::swap(dtypes_and_shapes_t,
            inserted_dtypes_and_shapes_[insert_dtypes_and_shapes_location_]);
  insert_dtypes_and_shapes_location_ =
      (insert_dtypes_and_shapes_location_ + 1) % max_timesteps_;

  buffer_.push_back(std::move(data));
  if (buffer_.size() < chunk_length_) return tensorflow::Status::OK();

  auto status = Finish();
  if (!status.ok()) {
    // Undo adding stuff to the buffer and undo the dtypes_and_shapes_ changes.
    buffer_.pop_back();
    insert_dtypes_and_shapes_location_ =
        PositiveModulo(insert_dtypes_and_shapes_location_ - 1, max_timesteps_);
    std::swap(dtypes_and_shapes_t,
              inserted_dtypes_and_shapes_[insert_dtypes_and_shapes_location_]);
  }
  return status;
}

tensorflow::Status Writer::AppendSequence(
    std::vector<tensorflow::Tensor> sequence) {
  if (sequence.empty()) {
    return tensorflow::errors::InvalidArgument(
        "AppendSequence called with empty data.");
  }
  for (int i = 0; i < sequence.size(); i++) {
    if (sequence[i].shape().dims() == 0) {
      return tensorflow::errors::InvalidArgument(
          "AppendSequence called with scalar tensor at index ", i, ".");
    }
    if (sequence[i].shape().dim_size(0) != sequence[0].shape().dim_size(0)) {
      return tensorflow::errors::InvalidArgument(
          "AppendSequence called with tensors of non equal batch dimension: ",
          internal::DtypesShapesString(sequence), ".");
    }
  }

  for (int i = 0; i < sequence[0].dim_size(0); i++) {
    std::vector<tensorflow::Tensor> step;
    step.reserve(sequence.size());
    for (const auto& column : sequence) {
      step.push_back(column.SubSlice(i));
    }
    TF_RETURN_IF_ERROR(Append(std::move(step)));
  }

  return tensorflow::Status::OK();
}

tensorflow::Status Writer::CreateItem(const std::string& table,
                                      int num_timesteps, double priority) {
  if (closed_) {
    return tensorflow::errors::FailedPrecondition(
        "Calling method CreateItem after Close has been called");
  }
  if (num_timesteps > chunks_.size() * chunk_length_ + buffer_.size()) {
    return tensorflow::errors::InvalidArgument(
        "Argument `num_timesteps` is larger than number of buffered "
        "timesteps.");
  }
  if (num_timesteps > max_timesteps_) {
    return tensorflow::errors::InvalidArgument(
        "`num_timesteps` must be <= `max_timesteps`");
  }

  const internal::DtypesAndShapes* dtypes_and_shapes = nullptr;
  TF_RETURN_IF_ERROR(GetFlatSignature(table, &dtypes_and_shapes));
  CHECK(dtypes_and_shapes != nullptr);
  if (dtypes_and_shapes->has_value()) {
    for (int t = 0; t < num_timesteps; ++t) {
      // Subtract 1 from the location since it is currently pointing to the next
      // write.
      const int check_offset = PositiveModulo(
          insert_dtypes_and_shapes_location_ - 1 - t, max_timesteps_);
      const auto& dtypes_and_shapes_t =
          inserted_dtypes_and_shapes_[check_offset];
      if (!dtypes_and_shapes_t.has_value()) {
        return tensorflow::errors::Internal(
            "Unexpected missing dtypes and shapes while calling CreateItem: "
            "expected a value at index ",
            check_offset, " (timestep offset ", t, ")");
      }

      if (dtypes_and_shapes_t->size() != (*dtypes_and_shapes)->size()) {
        return tensorflow::errors::InvalidArgument(
            "Unable to CreateItem to table ", table,
            " because Append was called with a tensor signature "
            "inconsistent with table signature.  Append for timestep "
            "offset ",
            t, " was called with ", dtypes_and_shapes_t->size(),
            " tensors, but table requires ", (*dtypes_and_shapes)->size(),
            " tensors per entry.  Table signature: ",
            internal::DtypesShapesString(**dtypes_and_shapes));
      }

      for (int c = 0; c < dtypes_and_shapes_t->size(); ++c) {
        const auto& signature_dtype_and_shape = (**dtypes_and_shapes)[c];
        const auto& seen_dtype_and_shape = (*dtypes_and_shapes_t)[c];
        if (seen_dtype_and_shape.dtype != signature_dtype_and_shape.dtype ||
            !signature_dtype_and_shape.shape.IsCompatibleWith(
                seen_dtype_and_shape.shape)) {
          return tensorflow::errors::InvalidArgument(
              "Unable to CreateItem to table ", table,
              " because Append was called with a tensor signature "
              "inconsistent with table signature.  Saw a tensor at "
              "timestep offset ",
              t, " in (flattened) tensor location ", c, " with dtype ",
              DataTypeString(seen_dtype_and_shape.dtype), " and shape ",
              seen_dtype_and_shape.shape.DebugString(),
              " but expected a tensor of dtype ",
              DataTypeString(signature_dtype_and_shape.dtype),
              " and shape compatible with ",
              signature_dtype_and_shape.shape.DebugString(),
              ".  (Flattened) table signature: ",
              internal::DtypesShapesString(**dtypes_and_shapes));
        }
      }
    }
  }

  int remaining = num_timesteps - buffer_.size();
  int num_chunks =
      remaining / chunk_length_ + (remaining % chunk_length_ ? 1 : 0);

  // Don't use additional chunks if the entire episode is contained in the
  // current buffer.
  if (remaining < 0) {
    num_chunks = 0;
  }

  PrioritizedItem item;
  item.set_key(NewID());
  item.set_table(table.data(), table.size());
  item.set_priority(priority);
  item.mutable_sequence_range()->set_length(num_timesteps);
  item.mutable_sequence_range()->set_offset(
      (chunk_length_ - (remaining % chunk_length_)) % chunk_length_);

  for (auto it = std::next(chunks_.begin(), chunks_.size() - num_chunks);
       it != chunks_.end(); it++) {
    item.add_chunk_keys(it->chunk_key());
  }
  if (!buffer_.empty()) {
    item.add_chunk_keys(next_chunk_key_);
  }

  pending_items_.push_back(item);

  if (buffer_.empty()) {
    auto status = WriteWithRetries();
    if (!status.ok()) pending_items_.pop_back();
    return status;
  }

  return tensorflow::Status::OK();
}

tensorflow::Status Writer::Flush() {
  if (closed_) {
    return tensorflow::errors::FailedPrecondition(
        "Calling method Flush after Close has been called");
  }

  if (!pending_items_.empty()) {
    return Finish();
  }
  if (!ConfirmItems(0)) {
    return tensorflow::errors::Internal(
        "Error when confirming that all items written to table.");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status Writer::StopItemConfirmationWorker() {
  // There is nothing to stop if there are no limitations on the number of
  // in flight items.
  if (!max_in_flight_items_.has_value()) return tensorflow::Status::OK();

  absl::MutexLock lock(&mu_);
  item_confirmation_worker_stop_requested_ = true;

  mu_.Await(absl::Condition(
      +[](bool* running) { return !(*running); },
      &item_confirmation_worker_running_));

  item_confirmation_worker_stop_requested_ = false;
  item_confirmation_worker_thread_ = nullptr;

  if (num_items_in_flight_ > 0) {
    return tensorflow::errors::DataLoss(
        "Item confirmation worker were stopped when ", num_items_in_flight_,
        " unconfirmed items (sent to server but validation response not yet "
        "received).");
  }
  num_items_in_flight_ = 0;
  return tensorflow::Status::OK();
}

void Writer::StartItemConfirmationWorker() {
  // Don't do anything if confirmations are not going to be sent anyway.
  if (!max_in_flight_items_.has_value()) return;

  absl::MutexLock lock(&mu_);

  // Sanity check the state of the writer. None of these should ever trigger in
  // the absence of fatal bugs.
  REVERB_CHECK(stream_ != nullptr);
  REVERB_CHECK(item_confirmation_worker_thread_ == nullptr);
  REVERB_CHECK_EQ(num_items_in_flight_, 0);
  REVERB_CHECK(!item_confirmation_worker_running_);
  REVERB_CHECK(!item_confirmation_worker_stop_requested_);

  item_confirmation_worker_thread_ = internal::StartThread(
      "WriterItemConfirmer",
      absl::bind_front(&Writer::ItemConfirmationWorker, this));
  mu_.Await(absl::Condition(
      +[](bool* started) { return *started; },
      &item_confirmation_worker_running_));
}

tensorflow::Status Writer::Close() {
  if (closed_) {
    return tensorflow::errors::FailedPrecondition(
        "Calling method Close after Close has been called");
  }
  if (!pending_items_.empty()) {
    TF_RETURN_IF_ERROR(Finish());
  }
  if (stream_) {
    stream_->WritesDone();
    REVERB_LOG_IF(REVERB_ERROR, !ConfirmItems(0))
        << "Unable to confirm that items were written.";
    TF_RETURN_IF_ERROR(StopItemConfirmationWorker());
    auto status = stream_->Finish();
    if (!status.ok()) {
      REVERB_LOG(REVERB_INFO) << "Received error when closing the stream: "
                << FormatGrpcStatus(status);
    }
    stream_ = nullptr;
  }
  chunks_.clear();
  closed_ = true;
  return tensorflow::Status::OK();
}

tensorflow::Status Writer::Finish() {
  std::vector<tensorflow::Tensor> batched_tensors;
  for (int i = 0; i < buffer_[0].size(); ++i) {
    std::vector<tensorflow::Tensor> tensors(buffer_.size());
    for (int j = 0; j < buffer_.size(); ++j) {
      const tensorflow::Tensor& item = buffer_[j][i];
      tensorflow::TensorShape shape = item.shape();
      shape.InsertDim(0, 1);
      // This should never fail due to dtype or shape differences, because the
      // dtype of tensors[j] is UNKNOWN and `shape` has the same number of
      // elements as `item`.
      REVERB_CHECK(tensors[j].CopyFrom(item, shape));
    }
    batched_tensors.emplace_back();
    TF_RETURN_IF_ERROR(
        tensorflow::tensor::Concat(tensors, &batched_tensors.back()));
  }

  ChunkData chunk;
  chunk.set_chunk_key(next_chunk_key_);
  chunk.mutable_sequence_range()->set_episode_id(episode_id_);
  chunk.mutable_sequence_range()->set_start(index_within_episode_);
  chunk.mutable_sequence_range()->set_end(index_within_episode_ +
                                          buffer_.size() - 1);

  if (delta_encoded_) {
    batched_tensors = DeltaEncodeList(batched_tensors, true);
    chunk.set_delta_encoded(true);
  }

  for (const auto& tensor : batched_tensors) {
    CompressTensorAsProto(tensor, chunk.add_data());
  }

  chunks_.push_back(std::move(chunk));

  auto status = WriteWithRetries();
  if (status.ok()) {
    index_within_episode_ += buffer_.size();
    buffer_.clear();
    next_chunk_key_ = NewID();
    while ((chunks_.size() - 1) * chunk_length_ >= max_timesteps_) {
      streamed_chunk_keys_.erase(chunks_.front().chunk_key());
      chunks_.pop_front();
    }
  } else {
    chunks_.pop_back();
  }
  return status;
}

tensorflow::Status Writer::WriteWithRetries() {
  while (true) {
    if (WritePendingData()) return tensorflow::Status::OK();
    stream_->WritesDone();
    TF_RETURN_IF_ERROR(StopItemConfirmationWorker());
    auto status = FromGrpcStatus(stream_->Finish());
    stream_ = nullptr;
    if (!tensorflow::errors::IsUnavailable(status)) return status;
  }
}

bool Writer::WritePendingData() {
  if (!stream_) {
    streamed_chunk_keys_.clear();
    context_ = absl::make_unique<grpc::ClientContext>();
    stream_ = stub_->InsertStream(context_.get());
    StartItemConfirmationWorker();
  }

  // Stream all chunks which are referenced by the pending items and haven't
  // already been sent. After the items has been inserted we want the server
  // to keep references only to the ones which the client still keeps
  // around.
  absl::flat_hash_set<uint64_t> item_chunk_keys;
  for (const auto& item : pending_items_) {
    for (uint64_t key : item.chunk_keys()) {
      item_chunk_keys.insert(key);
    }
  }
  std::vector<uint64_t> keep_chunk_keys;
  for (const ChunkData& chunk : chunks_) {
    if (item_chunk_keys.contains(chunk.chunk_key()) &&
        !streamed_chunk_keys_.contains(chunk.chunk_key())) {
      InsertStreamRequest request;
      request.set_allocated_chunk(const_cast<ChunkData*>(&chunk));
      grpc::WriteOptions options;
      options.set_no_compression();
      bool ok = stream_->Write(request, options);
      request.release_chunk();
      if (!ok) return false;
      streamed_chunk_keys_.insert(chunk.chunk_key());
    }
    if (streamed_chunk_keys_.contains(chunk.chunk_key())) {
      keep_chunk_keys.push_back(chunk.chunk_key());
    }
  }
  while (!pending_items_.empty()) {
    if (max_in_flight_items_.has_value() &&
        !ConfirmItems(max_in_flight_items_.value() - 1)) {
      return false;
    }
    InsertStreamRequest request;
    *request.mutable_item()->mutable_item() = pending_items_.front();
    *request.mutable_item()->mutable_keep_chunk_keys() = {
        keep_chunk_keys.begin(), keep_chunk_keys.end()};
    request.mutable_item()->set_send_confirmation(
        max_in_flight_items_.has_value());
    if (!stream_->Write(request)) return false;
    pending_items_.pop_front();
    if (request.item().send_confirmation()) {
      absl::MutexLock lock(&mu_);
      ++num_items_in_flight_;
    }
  }

  return true;
}

uint64_t Writer::NewID() {
  return absl::Uniform<uint64_t>(bit_gen_, 0, UINT64_MAX);
}

bool Writer::ConfirmItems(int limit) {
  absl::ReaderMutexLock lock(&mu_);
  auto done = [limit, this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return num_items_in_flight_ <= limit || !item_confirmation_worker_running_;
  };
  mu_.Await(absl::Condition(&done));
  return num_items_in_flight_ <= limit;
}

tensorflow::Status Writer::GetFlatSignature(
    absl::string_view table,
    const internal::DtypesAndShapes** dtypes_and_shapes) const {
  static const auto* empty_dtypes_and_shapes =
      new internal::DtypesAndShapes(absl::nullopt);
  if (!signatures_) {
    // No signatures available, return an unknown set.
    *dtypes_and_shapes = empty_dtypes_and_shapes;
    return tensorflow::Status::OK();
  }
  auto iter = signatures_->find(table);
  if (iter == signatures_->end()) {
    std::vector<std::string> table_names;
    for (const auto& table : *signatures_) {
      table_names.push_back(absl::StrCat("'", table.first, "'"));
    }
    return tensorflow::errors::InvalidArgument(
        "Unable to find signatures for table '", table,
        "' in signature cache.  Available tables: [",
        absl::StrJoin(table_names, ", "), "].");
  }
  *dtypes_and_shapes = &(iter->second);
  return tensorflow::Status::OK();
}

void Writer::ItemConfirmationWorker() {
  InsertStreamResponse response;
  while (true) {
    {
      absl::MutexLock lock(&mu_);

      // Setting this will unblock `StartItemConfirmationWorker` when Await is
      // called.
      item_confirmation_worker_running_ = true;

      // Wait until an item has been sent we expect the server to respond or
      // until the reader has been explicitly requested to stop prematurely.
      mu_.Await(absl::Condition(
          +[](Writer* w) ABSL_SHARED_LOCKS_REQUIRED(mu_) {
            return w->num_items_in_flight_ > 0 ||
                   w->item_confirmation_worker_stop_requested_;
          },
          this));
      if (item_confirmation_worker_stop_requested_) break;
    }
    if (!stream_->Read(&response)) break;
    absl::WriterMutexLock lock(&mu_);
    num_items_in_flight_--;
  }
  absl::WriterMutexLock lock(&mu_);
  item_confirmation_worker_running_ = false;
}

}  // namespace reverb
}  // namespace deepmind
