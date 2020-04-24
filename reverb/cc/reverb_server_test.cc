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

#include "reverb/cc/reverb_server.h"

#include <memory>

#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/platform/net.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace deepmind {
namespace reverb {
namespace {

TEST(ReverbServerTest, StartServer) {
  int port = internal::PickUnusedPortOrDie();
  std::unique_ptr<ReverbServer> server;
  TF_EXPECT_OK(ReverbServer::StartReverbServer(/*priority_tables=*/{},
                                               /*port=*/port, &server));
}

TEST(ReverbServerTest, ErrorOnUnavailablePort) {
  // We expect that port==-1 to always be unavailable.
  std::unique_ptr<ReverbServer> server;
  auto status = ReverbServer::StartReverbServer(/*priority_tables=*/{},
                                                /*port=*/-1, &server);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("Failed to BuildAndStart gRPC server"));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
