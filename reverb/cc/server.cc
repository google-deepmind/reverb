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

#include "reverb/cc/server.h"

#include <chrono>  // NOLINT(build/c++11) - grpc API requires it.
#include <csignal>
#include <memory>

#include "grpcpp/server_builder.h"
#include "absl/strings/str_cat.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/platform/grpc_utils.h"
#include "reverb/cc/platform/logging.h"

namespace deepmind {
namespace reverb {
namespace {

constexpr int kMaxMessageSize = 300 * 1000 * 1000;

}  // namespace

Server::Server(std::vector<std::shared_ptr<PriorityTable>> priority_tables,
               int port, std::shared_ptr<CheckpointerInterface> checkpointer)
    : port_(port) {

  reverb_service_ = absl::make_unique<ReverbServiceImpl>(
      std::move(priority_tables), std::move(checkpointer));

  server_ = grpc::ServerBuilder()
                .AddListeningPort(absl::StrCat("[::]:", port),
                                  MakeServerCredentials())
                .RegisterService(reverb_service_.get())
                .SetMaxSendMessageSize(kMaxMessageSize)
                .SetMaxReceiveMessageSize(kMaxMessageSize)
                .BuildAndStart();
}

tensorflow::Status Server::Initialize() {
  absl::WriterMutexLock lock(&mu_);
  REVERB_CHECK(!running_) << "Initialize() called twice?";
  if (!server_) {
    return tensorflow::errors::InvalidArgument(
        "Failed to BuildAndStart gRPC server");
  }
  running_ = true;
  REVERB_LOG(REVERB_INFO) << "Started replay server on port " << port_;
  return tensorflow::Status::OK();
}

/* static */ tensorflow::Status Server::StartServer(
    std::vector<std::shared_ptr<PriorityTable>> priority_tables, int port,
    std::unique_ptr<Server>* server) {
  // We can't use make_unique here since it can't access the private
  // Server constructor.
  std::unique_ptr<Server> s(new Server(std::move(priority_tables), port));
  TF_RETURN_IF_ERROR(s->Initialize());
  std::swap(s, *server);
  return tensorflow::Status::OK();
}

/* static */ tensorflow::Status Server::StartServer(
    std::vector<std::shared_ptr<PriorityTable>> priority_tables, int port,
    std::shared_ptr<CheckpointerInterface> checkpointer,
    std::unique_ptr<Server>* server) {
  // We can't use make_unique here since it can't access the private
  // Server constructor.
  std::unique_ptr<Server> s(
      new Server(std::move(priority_tables), port, std::move(checkpointer)));
  TF_RETURN_IF_ERROR(s->Initialize());
  std::swap(s, *server);
  return tensorflow::Status::OK();
}

Server::~Server() { Stop(); }

void Server::Stop() {
  absl::WriterMutexLock lock(&mu_);
  if (!running_) return;
  REVERB_LOG(REVERB_INFO) << "Shutting down replay server";

  // Closes the dependent services in the desirable order.
  reverb_service_->Close();

  // Set a deadline as the sampler streams never closes by themselves.
  server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));

  running_ = false;
}

void Server::Wait() {
  server_->Wait();
}

std::unique_ptr<Client> Server::InProcessClient() {
  grpc::ChannelArguments arguments;
  arguments.SetMaxReceiveMessageSize(kMaxMessageSize);
  arguments.SetMaxSendMessageSize(kMaxMessageSize);
  absl::WriterMutexLock lock(&mu_);
  return absl::make_unique<Client>(
      /* grpc_gen:: */ReverbService::NewStub(server_->InProcessChannel(arguments)));
}

}  // namespace reverb
}  // namespace deepmind
