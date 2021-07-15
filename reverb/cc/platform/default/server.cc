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

#include "reverb/cc/platform/server.h"

#include <chrono>  // NOLINT(build/c++11) - grpc API requires it.
#include <csignal>
#include <memory>

#include "grpcpp/server_builder.h"
#include "absl/strings/str_cat.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/client.h"
#include "reverb/cc/platform/grpc_utils.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/reverb_service_impl.h"
#include "reverb/cc/support/periodic_closure.h"

namespace deepmind {
namespace reverb {
namespace {

static std::function<void()> stop_server_fn = []() {};

void signal_handler(int signal) {
  REVERB_CHECK_EQ(signal, SIGINT);
  stop_server_fn();
}

class ServerImpl : public Server {
 public:
  ServerImpl(int port)
      : port_(port),
        signal_worker_(
            [this] {
              if (stop_signalled_) {
                Stop();
              }
            },
            absl::Milliseconds(250)) {}

  absl::Status Initialize(std::vector<std::shared_ptr<Table>> tables,
                          std::shared_ptr<Checkpointer> checkpointer) {
    absl::WriterMutexLock lock(&mu_);
    REVERB_CHECK(!running_) << "Initialize() called twice?";
    REVERB_RETURN_IF_ERROR(ReverbServiceImpl::Create(
        std::move(tables), std::move(checkpointer), &reverb_service_));
    server_ = grpc::ServerBuilder()
                  .AddListeningPort(absl::StrCat("[::]:", port_),
                                    MakeServerCredentials())
                  .RegisterService(reverb_service_.get())
                  .SetMaxSendMessageSize(kMaxMessageSize)
                  .SetMaxReceiveMessageSize(kMaxMessageSize)
                  .BuildAndStart();
    if (!server_) {
      return absl::InvalidArgumentError("Failed to BuildAndStart gRPC server");
    }
    running_ = true;
    REVERB_LOG(REVERB_INFO) << "Started replay server on port " << port_;
    REVERB_RETURN_IF_ERROR(signal_worker_.Start());
    return absl::OkStatus();
  }

  ~ServerImpl() override {
    signal_worker_.Stop().IgnoreError();
    Stop();
  }

  void Stop() override {
    absl::WriterMutexLock lock(&mu_);
    if (!running_) return;
    REVERB_LOG(REVERB_INFO) << "Shutting down replay server";

    reverb_service_->Close();

    // Set a deadline as the sampler streams never closes by themselves.
    server_->Shutdown(std::chrono::system_clock::now() +
                      std::chrono::seconds(5));

    running_ = false;
  }

  bool Wait() override {
    {
      absl::MutexLock lock(&mu_);
      if (!running_) return false;
    }

    // Register a signal handler for notifying the server about SIGINT signals.
    stop_server_fn = [server_ptr = this] { server_ptr->SignalStop(); };
    std::signal(SIGINT, signal_handler);

    server_->Wait();

    // Disable the signal handler by removing the callback.
    stop_server_fn = []() {};

    return stop_signalled_;
  }

  std::unique_ptr<Client> InProcessClient() override {
    grpc::ChannelArguments arguments;
    arguments.SetMaxReceiveMessageSize(kMaxMessageSize);
    arguments.SetMaxSendMessageSize(kMaxMessageSize);
    absl::WriterMutexLock lock(&mu_);
    return absl::make_unique<Client>(
        /* grpc_gen:: */ReverbService::NewStub(server_->InProcessChannel(arguments)));
  }

  std::string DebugString() const override {
    return absl::StrCat("Server(port=", port_,
                        ", reverb_service=", reverb_service_->DebugString(),
                        ")");
  }

  void SignalStop() { stop_signalled_ = true; }

  bool uses_async_service() const override { return false; }

 private:
  int port_;
  std::unique_ptr<ReverbServiceImpl> reverb_service_;
  std::unique_ptr<grpc::Server> server_ = nullptr;

  absl::Mutex mu_;
  bool running_ ABSL_GUARDED_BY(mu_) = false;

  // We can't call Stop directly from the signal handler as it requires mutex
  // locking which could result in deadlocks caused by recursive calls to the
  // the handler. We therefore use the indirect method of simply setting this
  // bool flag to true from the signal handler and deligate the actuall call
  // to Stop to a worker thread which periodically wakes up to check if Stop
  // should be called.
  bool stop_signalled_ = false;
  internal::PeriodicClosure signal_worker_;
};

}  // namespace

absl::Status StartServer(std::vector<std::shared_ptr<Table>> tables, int port,
                         std::shared_ptr<Checkpointer> checkpointer,
                         std::unique_ptr<Server> *server) {
  auto s = absl::make_unique<ServerImpl>(port);
  REVERB_RETURN_IF_ERROR(
      s->Initialize(std::move(tables), std::move(checkpointer)));
  *server = std::move(s);
  return absl::OkStatus();
}

}  // namespace reverb
}  // namespace deepmind
