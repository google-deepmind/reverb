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

#ifndef REVERB_CC_REVERB_SERVER_BIDI_REACTOR_H_
#define REVERB_CC_REVERB_SERVER_BIDI_REACTOR_H_

#include <queue>
#include <type_traits>

#include "grpcpp/grpcpp.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/task_worker.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/support/grpc_util.h"

namespace deepmind {
namespace reverb {

template <class Request, class Response, class TaskInfo, class TaskWorker>
class ReactorLock;

// Reactor implementing a bidirectional stream that inserts new tasks into
// a queue and lets background threads run the required callbacks.
// * Request and Response are the ones defined by the GRPC service.
// * TaskInfo is a class containing information about the task. The object must
//   include a callback that can be called by the TaskWorker.
// * TaskWorker is a class which executes tasks in the background and includes a
//   method:
//     `bool Schedule(TaskInfo task_info, TaskCallback task_callback)`
//   where TaskCallback is a function with the signature:
//       `void task_callback(TaskInfo task_info, const absl::Status&, bool
//           enough_queue_slots)`
//   See task_worker.h for more details.
//
// Note that writes to the stream have compression disabled. This reactor is
// supposed to send already compressed data (or very small messages).
template <class Request, class Response, class TaskInfo, class TaskWorker>
class ReverbServerBidiReactor
    : public grpc::ServerBidiReactor<Request, Response> {
  static_assert(std::is_base_of<InsertTaskInfo, TaskInfo>::value ||
                std::is_base_of<SampleTaskInfo, TaskInfo>::value ||
                std::is_base_of<InsertWorker, TaskWorker>::value ||
                std::is_base_of<SampleWorker, TaskWorker>::value,
                "Unsupported class for TaskInfo or TaskWorker");

 public:
  // Constructs a Reactor.
  // If allow_parallel_requests is false, new Reads will not be issued until the
  // current request has been processed. Otherwise, they will be issued once
  // OnReadDone is finished (unless reads are blocked because of lack of space
  // in the task worker).
  ReverbServerBidiReactor(bool allow_parallel_requests);

  // Starts a new read on the stream.
  void StartRead();

  /* Methods to be implemented by the specializations of the template. */
  // These methods are supposed to be called by the reactor callbacks.

  // Called in OnReadDone to validate the request and initialize reactor fields.
  virtual grpc::Status ProcessIncomingRequest(Request* request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Called in OnReadDone to decide if this request should produce a new task to
  // be scheduled in the TaskWorker.
  virtual bool ShouldScheduleFirstTask(const Request& request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Called in IsTaskCompleted to decide if a new task should be scheduled.
  virtual bool ShouldScheduleAnotherTask(const TaskInfo& task_info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Uses the information in the request to fill a TaskInfo object. Called in
  // OnReadDone.
  virtual grpc::Status FillTaskInfo(Request* request, TaskInfo* task_info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Uses the information in the old_tasks_info to fill a TaskInfo object.
  // Called in IsTaskCompleted.
  virtual TaskInfo FillTaskInfo(const TaskInfo& old_task_info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Gets the worker in order to schedule a new task.
  virtual TaskWorker* GetWorker();

  // Runs the callback from task_info and fills the vector with responses that
  // should be sent back. This is potentially large and it's called without
  // holding the Reactor's lock. Called by the TaskWorker.
  virtual absl::Status RunTaskAndFillResponses(std::vector<Response>* responses,
                                               const TaskInfo& task_info);

  // Called when running the task to decide if it should retry the task after
  // getting a timeout. Called by the task scheduled in InsertNewTask.
  virtual bool ShouldRetryOnTimeout(const TaskInfo& task_info);

  /* Reactor utils */

  // Returns the number of pending responses that have to be sent. It should
  // always be called with the mutex held.
  // Note that ABSL annotations won't work because they are called in
  // implementations of the template without access to the mutex.
  int64_t NumPendingResponses();

  /* Reactor callbacks. */

  // Reads new requests.
  // If HalfClose, cancellation or failure happens, finalizes the stream.
  // Otherwise, it processes the request and:
  // * If the request includes new tasks, it schedules them in the TaskWorker.
  // * If the TaskQueue is not almost full, and if the reactor allows to process
  //   multiple requests in parallel (or the current request has been processed
  //   already), it starts a new read on the stream.
  void OnReadDone(bool ok) override;

  // After writing successfully a message in the stream, checks if there are
  // more messages to be sent and if it should reschedule a new read.
  // If there is any error, it finishes the stream.
  void OnWriteDone(bool ok) override;

  // Waits for pending tasks to finish and finalizes the reactor.
  void OnDone() override;

  // Sets the reactor as cancelled.
  void OnCancel() override;

 private:
  friend ReactorLock<Request, Response, TaskInfo, TaskWorker>;

  // Tells whether reactor should be deleted (upon the second call of this
  // method). One call is done by OnDone callback, second by
  // SetReactorAsFinished (order depends on the thread timing in GRPC).
  // Reactor has to be freed after both finish.
  bool ShouldDeleteReactor() ABSL_EXCLUSIVE_LOCKS_REQUIRED(seq_mu_);

  // Finishes the reactor. It fails if any of the conditions to finish the
  // reactor doesn't hold. The conditions are:
  //   * The reactor cannot be already set as finished.
  //   * If status is Ok, there shouldn't be pending tasks.
  void SetReactorAsFinished(grpc::Status status)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns true if the conditions to finish the reactor are met:
  // * The reactor is not already finished.
  // * The reactor is not still waiting for more requests.
  // * THere are no pending tasks.
  bool ShouldFinish() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Schedules a task in the TaskQueue using the information in the request.
  grpc::Status ScheduleFirstTask() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Schedules a new task in the TaskQueue after processing a task.
  void ScheduleAnotherTask(const TaskInfo& task_info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Inserts a new task in the TaskQueue.
  void InsertNewTask(TaskInfo task_info) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Checks if task is completed and updates the reactor information in such
  // case. Task should be retried when false is returned.
  bool IsTaskCompleted(std::vector<Response> responses,
                       const TaskInfo& task_info, const absl::Status& status,
                       bool enough_queue_slots, bool already_called)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Returns true if current reads are blocked because the queue was full,
  // and it can be unblocked.
  bool ShouldUnblockReads(bool enough_queue_slots)
      ABSL_SHARED_LOCKS_REQUIRED(mu_);

  // Returns true if there are no more pending tasks or items to be sent for
  // the current request.
  bool IsCurrentRequestProcessed() ABSL_SHARED_LOCKS_REQUIRED(mu_);

  // Returns true if the reactor can still receive new reads.
  bool IsReadStillPossible() ABSL_SHARED_LOCKS_REQUIRED(mu_);

  // Returns true if the reactor is ready to process a new request.
  bool IsReadyForNewRead() ABSL_SHARED_LOCKS_REQUIRED(mu_);

  // Incoming messages are handled one at a time. That is StartRead is not
  // called until `request_` has been completely salvaged. Fields accessed
  // only by OnRead are thus thread safe and require no additional mutex to
  // control access.
  //
  // The following fields are ONLY accessed by OnRead (and subcalls):
  //  - request_
  //

  // Message where new requests are unpacked to.
  Request request_;

  absl::Mutex mu_ ABSL_ACQUIRED_BEFORE(seq_mu_);

  // Each insert where confirmation is requested will push a response to this
  // queue in its callback. Responses are written, one at a time, in the same
  // order as they are pushed into the queue. Note that since strict ordering
  // is enforced through sequence numbering, the order of the responses will
  // exactly match that of the requests.
  std::queue<Response> responses_to_send_ ABSL_GUARDED_BY(mu_);

  // The number of tasks scheduled but not yet completed. Note that even though
  // we enforce strict ordering, the reactor is allowed to run ahead and
  // schedule multiple tasks to the queue.
  //
  // If the reactor is prematurely closed (i.e encountered an error), there
  // might still be tasks in the worker queue that waiting for their turn.
  // The callback of these tasks directly reference the Reactor-object and it
  // is therefore unsafe to destroy the reactor until this count reaches zero.
  // `OnDone` enforces this condition.
  int64_t num_tasks_pending_completion_ ABSL_GUARDED_BY(mu_);

  // When false, it means that the client has notified that it is not writing
  // anymore, or that the stream has been finished/cancelled.
  bool still_reading_ ABSL_GUARDED_BY(mu_);

  // Once this is true, we cannot write or read. It is used to signal that
  // Finish has been called.
  bool is_finished_ ABSL_GUARDED_BY(mu_);

  // Once this is true, pending tasks will be discarded.
  bool is_cancelled_ ABSL_GUARDED_BY(mu_);

  // Was ShouldDeleteReactor already called. Reusing seq_mu_ lock
  // to not cause performance issues between different locks in the future
  // (L1 cache flushing etc.).
  bool should_delete_called_ ABSL_GUARDED_BY(seq_mu_) = false;

  // If true, the reactor is not reading and it is expecting a pending
  // task to complete and start the next Read.
  bool is_read_blocked_by_full_queue_ ABSL_GUARDED_BY(mu_);

  // If true, the reactor will try to start a new read in OnReadDone. Otherwise,
  // it waits until the current request is fully handled (new reads might be
  // started in OnWriteDone or IsTaskCompleted).
  const bool allow_parallel_requests_ ABSL_GUARDED_BY(mu_);


  // Sequence numbers are used to guarantee that the tasks are scheduled
  // preserving the request order. Each new task will get a sequence
  // number assigned, and won't run until all previous tasks are done.

  // TODO(b/184030177): Ideally the sequence numbers are per table.

  // Sequence number to assign to the next item.
  int64_t request_seq_num_ ABSL_GUARDED_BY(mu_) = 0;

  // Sequence number of the last task executed. It uses a different
  // lock so the threads waiting to run a task don't block on the reactor
  // updating other reactor values.
  int64_t last_seq_num_finished_ ABSL_GUARDED_BY(seq_mu_) = -1;
  absl::Mutex seq_mu_;
};

// ReactorLock is used for deleting Reactor after releasing a lock (deletion
// of the Reactor is conditioned upon the state protected by the lock).
template <class Request, class Response, class TaskInfo, class TaskWorker>
class ABSL_SCOPED_LOCKABLE ReactorLock {
 public:
  explicit ReactorLock(ReverbServerBidiReactor<Request, Response, TaskInfo,
      TaskWorker> *reactor) ABSL_EXCLUSIVE_LOCK_FUNCTION(reactor->mu_)
      : reactor_(reactor) {
    reactor_->mu_.Lock();
  }

  ReactorLock(const ReactorLock &) = delete;  // NOLINT(runtime/mutex)
  ReactorLock(ReactorLock&&) = delete;  // NOLINT(runtime/mutex)
  ReactorLock& operator=(const ReactorLock&) = delete;
  ReactorLock& operator=(ReactorLock&&) = delete;

  void FinishReactor(grpc::Status status) {
    reactor_->SetReactorAsFinished(status);
    bool should_delete;
    {
      absl::MutexLock lock(&reactor_->seq_mu_);
      should_delete = reactor_->ShouldDeleteReactor();
      // Unlock mu_ before seq_mu_, so that in case Reactor is deleted by OnDone
      // we don't reference Reactor after deletion.
      reactor_->mu_.Unlock();
    }
    if (should_delete) {
      delete reactor_;
    }
    // We already released Reactor's mu_ mutex, don't release again in the
    // destructor.
    reactor_ = nullptr;
  }

  ~ReactorLock() ABSL_UNLOCK_FUNCTION() {
    if (reactor_) {
      reactor_->mu_.Unlock();
    }
  }

 private:
  ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>* reactor_;
};

/*****************************************************************************
 * Implementation of the template                                            *
 *****************************************************************************/
template <class Request, class Response, class TaskInfo, class TaskWorker>
ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    ReverbServerBidiReactor(bool allow_parallel_requests)
    : num_tasks_pending_completion_(0),
      still_reading_(true),
      is_finished_(false),
      is_cancelled_(false),
      is_read_blocked_by_full_queue_(false),
      allow_parallel_requests_(allow_parallel_requests),
      request_seq_num_(0),
      last_seq_num_finished_(-1) {}

template <class Request, class Response, class TaskInfo, class TaskWorker>
void ReverbServerBidiReactor<Request, Response, TaskInfo,
                             TaskWorker>::StartRead() {
  grpc::ServerBidiReactor<Request, Response>::StartRead(&request_);
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
grpc::Status
ReverbServerBidiReactor<Request, Response, TaskInfo,
                        TaskWorker>::ProcessIncomingRequest(Request* request)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                      "ProcessingIncomingRequest is not implemented");
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
bool ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    ShouldScheduleFirstTask(const Request& request)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  // Since the template cannot have pure virtual methods, we fail here to
  // force instantiations to re-define this method.
  REVERB_QCHECK(false) << "The implementation of the ReverbServerBidiReactor "
                          "has to override ShouldScheduleFirstTask.";
  return false;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
bool ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    ShouldScheduleAnotherTask(const TaskInfo& task_info)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  // Since the template cannot have pure virtual methods, we fail here to
  // force instantiations to re-define this method.
  REVERB_QCHECK(false) << "The implementation of the ReverbServerBidiReactor "
                          "has to override ShouldScheduleAnotherTask.";
  return false;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
grpc::Status
ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::FillTaskInfo(
    Request* request, TaskInfo* task_info) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                      "FillTaskInfo is not implemented");
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
TaskInfo ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    FillTaskInfo(const TaskInfo& old_task_info)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  // If used, this method has to be re-implemented by the implementations of
  // the template
  REVERB_QCHECK(false) << "The implementation of the ReverbServerBidiReactor "
                          "has to override FillTaskInfo.";
  return old_task_info;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
TaskWorker*
ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::GetWorker()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  REVERB_LOG(REVERB_ERROR) << "GetWorker is not implemented.";
  return nullptr;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
absl::Status ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    RunTaskAndFillResponses(std::vector<Response>* responses,
                            const TaskInfo& task_info) {
  return absl::UnimplementedError("RunTaskAndFillResponses is not implemented");
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
bool ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    ShouldRetryOnTimeout(const TaskInfo& task_info) {
  return false;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
int64_t ReverbServerBidiReactor<Request, Response, TaskInfo,
                            TaskWorker>::NumPendingResponses() {
  return responses_to_send_.size();
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
void ReverbServerBidiReactor<Request, Response, TaskInfo,
                             TaskWorker>::OnReadDone(bool ok) {
  // Read until the client sends a HalfClose or the stream is cancelled.
  ReactorLock<Request, Response, TaskInfo, TaskWorker> lock(this);

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
      lock.FinishReactor(grpc::Status::OK);
      return;
    }
    return;
  }

  auto status = ProcessIncomingRequest(&request_);
  if (!status.ok()) {
    lock.FinishReactor(status);
    return;
  }

  if (ShouldScheduleFirstTask(request_)) {
    status = ScheduleFirstTask();
    if (!status.ok()) {
      lock.FinishReactor(status);
      return;
    }
  }

  // If the Worker is almost full, the next OnRead could block
  // on the call to `Worker::Schedule (that waits for the queue to be
  // non-full. In order to avoid blocking the gRPC threads, we delay the
  // start of the next read until the worker has more space.
  // If we don't allow parallel requests, but there are no pending tasks for
  // this request, we also re-enable Reads.
  if (!is_read_blocked_by_full_queue_ && IsReadyForNewRead()) {
    grpc::ServerBidiReactor<Request, Response>::StartRead(&request_);
  }
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
void ReverbServerBidiReactor<Request, Response, TaskInfo,
                             TaskWorker>::OnWriteDone(bool ok) {
  ReactorLock<Request, Response, TaskInfo, TaskWorker> lock(this);
  if (is_finished_) {
    REVERB_LOG(REVERB_ERROR)
        << "OnWriteDone was called after the reactor was finished";
    return;
  }
  if (!ok) {
    // A half close has been received and thus there will be no more reads.
    auto status = grpc::Status(
        grpc::StatusCode::INTERNAL,
        "Error when sending response (the stream is being closed).");
    lock.FinishReactor(status);
    return;
  }
  // Message was successfully sent.
  responses_to_send_.pop();

  // If there are more responses queued up then start sending the next
  // message straight away. If the queue is empty then future writes will
  // be started by OnTaskComplete.
  if (!responses_to_send_.empty()) {
    grpc::WriteOptions options;
    options.set_no_compression();
    grpc::ServerBidiReactor<Request, Response>::StartWrite(
        &responses_to_send_.front(), options);
    return;
  }
  // If there are no more responses to send, we are in a single-request mode
  // (i.e., we don't handle a new request until we have finished the
  // previous one), and there are no pending tasks from the current request,
  // we can start a new read (as long as the reactor is not finished).
  if (!allow_parallel_requests_ && IsReadyForNewRead()) {
    grpc::ServerBidiReactor<Request, Response>::StartRead(&request_);

    return;
  }

  // There are no pending writes so if we are no longer reading from the
  // stream and there are no pending tasks then we are done.
  if (!still_reading_ && num_tasks_pending_completion_ == 0) {
    lock.FinishReactor(grpc::Status::OK);
    return;
  }
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
void ReverbServerBidiReactor<Request, Response, TaskInfo,
                             TaskWorker>::OnDone() {
  bool should_delete_;
  {
    absl::MutexLock lock(&seq_mu_);
    should_delete_ = ShouldDeleteReactor();
  }
  if (should_delete_) {
    delete this;
  }
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
void ReverbServerBidiReactor<Request, Response, TaskInfo,
                             TaskWorker>::OnCancel() {
  absl::MutexLock lock(&mu_);
  still_reading_ = false;
  is_cancelled_ = true;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
bool ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    ShouldDeleteReactor() ABSL_EXCLUSIVE_LOCKS_REQUIRED(seq_mu_) {
  if (!should_delete_called_) {
    should_delete_called_ = true;
    return false;
  }
  return true;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
void ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    SetReactorAsFinished(grpc::Status status)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  REVERB_CHECK(!is_finished_);

  // Sanity check that everything has been completed when the reactor is
  // closed with a successful status.
  REVERB_CHECK(
      (responses_to_send_.empty() && num_tasks_pending_completion_ == 0) ||
      !status.ok());

  // Once the reactor is finished, we won't send any more responses.
  std::queue<Response>().swap(responses_to_send_);
  is_finished_ = true;
  grpc::ServerBidiReactor<Request, Response>::Finish(status);
  still_reading_ = false;
  // We don't wait for responses_to_send to be empty because once OnFinish
  // is called, we will not send more confirmations and OnFinish is the last
  // call made on a stream.
  auto is_ready_for_deletion = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return num_tasks_pending_completion_ == 0;
  };
  mu_.Await(absl::Condition(&is_ready_for_deletion));
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
bool ReverbServerBidiReactor<Request, Response, TaskInfo,
                             TaskWorker>::ShouldFinish()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return responses_to_send_.empty() && num_tasks_pending_completion_ == 0 &&
         !still_reading_ && !is_finished_;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
grpc::Status ReverbServerBidiReactor<Request, Response, TaskInfo,
                                     TaskWorker>::ScheduleFirstTask()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TaskInfo task_info;
  if (auto status = FillTaskInfo(&request_, &task_info); !status.ok()){
    return status;
  }
  InsertNewTask(std::move(task_info));
  return grpc::Status::OK;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
void ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    ScheduleAnotherTask(const TaskInfo& task_info)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TaskInfo new_task_info = FillTaskInfo(task_info);
  InsertNewTask(std::move(new_task_info));
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
void ReverbServerBidiReactor<Request, Response, TaskInfo,
                             TaskWorker>::InsertNewTask(TaskInfo task_info)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TaskWorker* worker = GetWorker();
  if (worker == nullptr){
    REVERB_CHECK(false) << "Task workers not initialized.";
  }
  num_tasks_pending_completion_++;

  int64_t seq_num = request_seq_num_++;

  bool queue_is_not_full = worker->Schedule(
      std::move(task_info),
      [this, seq_num](TaskInfo task_info, absl::Status status,
                      bool enough_queue_slots) {
        // Block until the callback of the previous task (from this reactor)
        // has been triggered. This ensures that order within a single
        // stream is maintained even if multiple workers are used.
        {
          absl::MutexLock lock(&seq_mu_);
          auto is_first_in_queue =
              [this, seq_num]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(seq_mu_) {
                return (seq_num == (last_seq_num_finished_ + 1));
              };
          seq_mu_.Await(absl::Condition(&is_first_in_queue));
        }
        std::vector<Response> responses;
        bool already_called = false;
        while (!IsTaskCompleted(responses, task_info, status,
                                enough_queue_slots, already_called)) {
          // RunTaskAndFillResponses can block for a long time
          // (due to the RateLimiter) so it is executed without holding
          // any mutexes (to avoid blocking a gRPC thread).
          status = RunTaskAndFillResponses(&responses, task_info);
          already_called = true;
        }
      });

  if (!queue_is_not_full) {
    is_read_blocked_by_full_queue_ = true;
  }
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
bool ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    IsTaskCompleted(std::vector<Response> responses, const TaskInfo& task_info,
                    const absl::Status& status, bool enough_queue_slots,
                    bool already_called) {
  ReactorLock<Request, Response, TaskInfo, TaskWorker> lock(this);
  if (!is_cancelled_ && !is_finished_) {
    if (!already_called && status.ok()) {
      // Not yet executed and no prior errors.
      return false;
    }
    if (absl::IsDeadlineExceeded(status) && ShouldRetryOnTimeout(task_info)) {
      // If the timeout has exceeded but the stream is still alive then
      // we simply try again if retries on timeout are possible for this task.
      return false;
    }
  }
  num_tasks_pending_completion_--;
  {
    absl::MutexLock seq_lock(&seq_mu_);
    last_seq_num_finished_++;
  }
  // In the event of an error, the reactor will still purge the queue and
  // IsTaskCompleted will be triggered even though Finish has been called.
  // When this happen we just stop early.
  if (is_finished_) {
    REVERB_LOG_IF(REVERB_WARNING, !status.ok())
        << "Ignoring error as reactor already closed with previous error. "
           "Ignored status ("
        << task_info.DebugString() << "): " << status.ToString();
    return true;
  }

  // If the reactor was not finished and the task failed, we use it to close
  // the stream.
  if (!status.ok() || !already_called) {
    grpc::Status grpc_status;
    if (status.ok() || (absl::IsDeadlineExceeded(status) && is_cancelled_)) {
      // When the timeout is exceeded we check if the reactor has been
      // cancelled concurrently with the attempted InsertOrAssign call.
      // If this is the case then we replace the error with the error
      // which is used for tasks that are popped from the queue after
      // the flag has been set.
      grpc_status =
          ToGrpcStatus(absl::CancelledError("Stream has been cancelled"));
    } else {
      grpc_status = ToGrpcStatus(status);
    }
    lock.FinishReactor(grpc_status);
    return true;
  }
  // If the queue was empty before we add new responses, we need to write the
  // first response. Afterwards, OnWriteDone will trigger the next writes
  // automatically.
  bool should_start_sending = responses_to_send_.empty() && !responses.empty();
  for (auto r : responses) {
    responses_to_send_.push(std::move(r));
  }

  // If the queue was non empty before the new message was added then the
  // write will eventually be triggered automatically by OnWriteDone.
  if (should_start_sending) {
    grpc::WriteOptions options;
    options.set_no_compression();  // Data is already compressed.
    grpc::ServerBidiReactor<Request, Response>::StartWrite(
        &responses_to_send_.front(), options);
  }

  if (ShouldFinish()) {
    lock.FinishReactor(grpc::Status::OK);
    return true;
  }

  if (ShouldScheduleAnotherTask(task_info)) {
    ScheduleAnotherTask(task_info);
  }

  if (ShouldUnblockReads(enough_queue_slots)) {
    is_read_blocked_by_full_queue_ = false;
    grpc::ServerBidiReactor<Request, Response>::StartRead(&request_);
  }
  return true;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
bool ReverbServerBidiReactor<Request, Response, TaskInfo,
                             TaskWorker>::IsCurrentRequestProcessed()
    ABSL_SHARED_LOCKS_REQUIRED(mu_) {
  return (num_tasks_pending_completion_ == 0) && (responses_to_send_.empty());
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
bool ReverbServerBidiReactor<Request, Response, TaskInfo,
                             TaskWorker>::IsReadyForNewRead()
    ABSL_SHARED_LOCKS_REQUIRED(mu_) {
  if (!IsReadStillPossible()) {
    return false;
  }
  if (allow_parallel_requests_) {
    // When we allow parallel requests, we can start a new read at any point (as
    // long as the previous read was handled).
    return true;
  }
  return IsCurrentRequestProcessed();
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
bool ReverbServerBidiReactor<Request, Response, TaskInfo, TaskWorker>::
    ShouldUnblockReads(bool enough_queue_slots)
        ABSL_SHARED_LOCKS_REQUIRED(mu_) {
  // We can't resume reads if we were never blocked to begin with.
  if (!is_read_blocked_by_full_queue_) {
    return false;
  }
  // We can't unblock reads based on queue space if only one request can
  // be active and the current request is not yet processed.
  if (!IsReadyForNewRead()) {
    return false;
  }
  if (!IsReadStillPossible()) {
    return false;
  }

  // It is possible for the queue to be full even though there are no
  // pending tasks in the reactor. This if there are a large number of
  // concurrent connections (and thus reactors), or there is a very large
  // number of insertions being scheduled from one (or more) of the other
  // reactors. When this happen we need to resume the read (even though the
  // queue is full) since it is the last chance for the reactor to do so.
  return enough_queue_slots || num_tasks_pending_completion_ == 0;
}

template <class Request, class Response, class TaskInfo, class TaskWorker>
bool ReverbServerBidiReactor<Request, Response, TaskInfo,
                             TaskWorker>::IsReadStillPossible()
    ABSL_SHARED_LOCKS_REQUIRED(mu_) {
  // If half close has been received or if an error has been encountered
  // then we should not resume reading.
  if (!still_reading_ || is_cancelled_ || is_finished_) {
    return false;
  }
  return true;
}

}  // namespace reverb
}  // namespace deepmind
#endif  // REVERB_CC_REVERB_SERVER_BIDI_REACTOR_H_
