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

#ifndef REVERB_CC_REVERB_SERVER_TABLE_REACTOR_H_
#define REVERB_CC_REVERB_SERVER_TABLE_REACTOR_H_

#include <queue>
#include <type_traits>

#include "grpcpp/grpcpp.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/task_worker.h"

namespace deepmind {
namespace reverb {
#define GRPC_CALL_AND_RETURN_IF_ERROR(expr, func) \
  if (auto status = expr; !status.ok()) {         \
    func(status);                                 \
    return;                                       \
  }

// Reactor implementing a bidirectional stream that enqueues work onto
// appropriate tables.
// * Request and Response are the ones defined by the GRPC service.
//
// Note that writes to the stream have compression disabled. This reactor is
// supposed to send already compressed data (or very small messages).
template <class Request, class Response, class ResponseCtx>
class ReverbServerTableReactor
    : public grpc::ServerBidiReactor<Request, Response> {

 protected:
  // Called in OnReadDone to process incoming request.
  virtual grpc::Status ProcessIncomingRequest(Request* request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  /* Reactor callbacks. */

  // Reads new requests.
  // If HalfClose, cancellation or failure happens, finalizes the stream.
  // Otherwise, it processes the request with ProcessIncomingRequest.
  // It is request handler's responsibility to call MaybeStartRead to start
  // handling another request.
  void OnReadDone(bool ok) override;

  // After writing successfully a message in the stream, checks if there are
  // more messages to be sent.
  // If there is any error, it finishes the stream.
  void OnWriteDone(bool ok) override;

  // Waits for pending tasks to finish and finalizes the reactor.
  void OnDone() override;

  // Sets the reactor as cancelled.
  void OnCancel() override;

  // Schedules another read if possible. Should be called after OnReadDone.
  void MaybeStartRead();

  // Finishes the reactor. It fails if any of the conditions to finish the
  // reactor doesn't hold. The conditions are:
  //   * The reactor cannot be already set as finished.
  //   * If status is Ok, there shouldn't be pending tasks.
  void SetReactorAsFinished(grpc::Status status)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns true if the conditions to finish the reactor are met:
  // * The reactor is not already finished.
  // * The reactor is not still waiting for more requests.
  // * There are no pending tasks.
  bool ShouldFinish() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Starts sending another queued response to the client (if available).
  void MaybeSendNextResponse() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

 protected:
  // Incoming messages are handled one at a time. That is StartRead is not
  // called until `request_` has been completely salvaged. Fields accessed
  // only by OnRead are thus thread safe and require no additional mutex to
  // control access.

  // Message where new requests are unpacked to.
  Request request_;

  absl::Mutex mu_;

  // Queued responses to be sent to the client.
  std::queue<ResponseCtx> responses_to_send_ ABSL_GUARDED_BY(mu_);

  // When false, it means that the client has notified that it is not writing
  // anymore, or that the stream has been finished/cancelled.
  bool still_reading_ ABSL_GUARDED_BY(mu_) = true;

  // Once this is true, we cannot write or read. It is used to signal that
  // Finish has been called.
  bool is_finished_ ABSL_GUARDED_BY(mu_) = false;

  // Is reactor cancelled by the client. Once this is true,
  // pending tasks will be discarded.
  bool is_cancelled_ ABSL_GUARDED_BY(mu_) = false;
};

/*****************************************************************************
 * Implementation of the template                                            *
 *****************************************************************************/
template <class Request, class Response, class ResponseCtx>
void ReverbServerTableReactor<Request, Response,
                              ResponseCtx>::MaybeSendNextResponse() {
  if (responses_to_send_.empty() || is_finished_) {
    return;
  }
  grpc::WriteOptions options;
  options.set_no_compression();
  grpc::ServerBidiReactor<Request, Response>::StartWrite(
      &responses_to_send_.front().payload, options);
}

template <class Request, class Response, class ResponseCtx>
void ReverbServerTableReactor<Request, Response, ResponseCtx>::OnReadDone(
    bool ok) {
  // Read until the client sends a HalfClose or the stream is cancelled.
  absl::MutexLock lock(&mu_);

  if (!ok || is_finished_) {
    // A half close has been received and thus there will be no more reads.
    // Or SetReactorAsFinished was called before we acquire the lock (it
    // can be called by the TaskWorkers, outside of the reactor).
    still_reading_ = false;

    // Close the reactor iff all pending tasks are done AND all pending
    // responses have been sent. If all conditions are fulfilled then
    // everything went according to plan and we close the reactor with a
    // successful status.
    if (ShouldFinish()) {
      SetReactorAsFinished(grpc::Status::OK);
    }
    return;
  }

  GRPC_CALL_AND_RETURN_IF_ERROR(ProcessIncomingRequest(&request_),
                                SetReactorAsFinished);
}

template <class Request, class Response, class ResponseCtx>
void ReverbServerTableReactor<Request, Response, ResponseCtx>::OnWriteDone(
    bool ok) {
  absl::MutexLock lock(&mu_);
  if (is_finished_) {
    REVERB_LOG(REVERB_ERROR)
        << "OnWriteDone was called after the reactor was finished";
    return;
  }
  if (!ok) {
    // No more reads can happen after this point.
    auto status = grpc::Status(
        grpc::StatusCode::INTERNAL,
        "Error when sending response (the stream is being closed).");
    SetReactorAsFinished(status);
    return;
  }
  // Message was successfully sent.
  responses_to_send_.pop();

  // There are no pending writes so if we are no longer reading from the
  // stream and there are no pending tasks then we are done.
  if (!still_reading_ && responses_to_send_.empty()) {
    SetReactorAsFinished(grpc::Status::OK);
    return;
  }
  MaybeSendNextResponse();
}

template <class Request, class Response, class ResponseCtx>
void ReverbServerTableReactor<Request, Response, ResponseCtx>::OnDone() {
  {
    absl::MutexLock lock(&mu_);
    still_reading_ = false;
    REVERB_CHECK(is_finished_);
  }
  delete this;
}

template <class Request, class Response, class ResponseCtx>
void ReverbServerTableReactor<Request, Response, ResponseCtx>::OnCancel() {
  absl::MutexLock lock(&mu_);
  still_reading_ = false;
  is_cancelled_ = true;
}

template <class Request, class Response, class ResponseCtx>
void ReverbServerTableReactor<Request, Response,
                              ResponseCtx>::MaybeStartRead() {
  absl::MutexLock lock(&mu_);
  if (still_reading_ && !is_cancelled_ && !is_finished_) {
    grpc::ServerBidiReactor<Request, Response>::StartRead(&request_);
  }
}

template <class Request, class Response, class ResponseCtx>
void ReverbServerTableReactor<
    Request, Response, ResponseCtx>::SetReactorAsFinished(grpc::Status status)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  REVERB_CHECK(!is_finished_);

  // Sanity check that everything has been completed when the reactor is
  // closed with a successful status.
  REVERB_CHECK(responses_to_send_.empty() || !status.ok());

  // Once the reactor is finished, we won't send any more responses.
  std::queue<ResponseCtx>().swap(responses_to_send_);
  is_finished_ = true;
  grpc::ServerBidiReactor<Request, Response>::Finish(status);
}

template <class Request, class Response, class ResponseCtx>
bool ReverbServerTableReactor<Request, Response, ResponseCtx>::ShouldFinish()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return responses_to_send_.empty() && !still_reading_ && !is_finished_;
}
}  // namespace reverb
}  // namespace deepmind
#endif  // REVERB_CC_REVERB_SERVER_TABLE_REACTOR_H_
