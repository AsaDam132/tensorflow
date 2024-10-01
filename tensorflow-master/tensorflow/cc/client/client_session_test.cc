/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/cc/client/client_session.h"

#include <utility>
#include <vector>

#include "absl/synchronization/barrier.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/threadpool_options.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace {

using ops::Add;
using ops::BatchMatMul;
using ops::Const;
using ops::Mul;
using ops::Placeholder;
using ops::Sub;

tensorflow::SessionOptions GetSessionOptions() {
  tensorflow::SessionOptions options;
  // Disable optimizations for static graph to allow calls to Session::Extend.
  options.config.mutable_experimental()->set_disable_optimize_for_static_graph(
      true);
  return options;
}

class CustomThreadPoolImpl : public thread::ThreadPoolInterface {
 public:
  explicit CustomThreadPoolImpl(int numThreads) {
    underlying_threadpool_.reset(new thread::ThreadPool(
        tensorflow::Env::Default(), "custom_threadpool", numThreads));
    num_schedule_called_ = 0;
  }

  void Schedule(std::function<void()> fn) override {
    num_schedule_called_ += 1;
    underlying_threadpool_->Schedule(std::move(fn));
  }

  void ScheduleWithHint(std::function<void()> fn, int start, int end) override {
    num_schedule_called_ += 1;
    underlying_threadpool_->ScheduleWithHint(std::move(fn), start, end);
  }

  void Cancel() override {}

  int NumThreads() const override {
    return underlying_threadpool_->NumThreads();
  }

  int CurrentThreadId() const override {
    return underlying_threadpool_->CurrentThreadId();
  }

  int GetNumScheduleCalled() { return num_schedule_called_; }

 private:
  int num_schedule_called_;
  std::unique_ptr<tensorflow::thread::ThreadPool> underlying_threadpool_;
};

TEST(ClientSessionTest, Basic) {
  Scope root = Scope::NewRootScope();
  auto c = Const(root, {{1, 1}});
  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run({c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({1, 1}, {1, 2}));
}

TEST(ClientSessionTest, Feed) {
  Scope root = Scope::NewRootScope();
  auto a = Placeholder(root, DT_INT32);
  auto b = Placeholder(root, DT_INT32);
  auto c = Add(root, a, b);
  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run({{a, 1}, {b, 41}}, {c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({42}, {}));
}

TEST(ClientSessionTest, Extend) {
  Scope root = Scope::NewRootScope();
  auto a = Placeholder(root, DT_INT32, Placeholder::Shape({2}));
  auto c = Add(root, a, {2, 2});
  ClientSession session(root, GetSessionOptions());
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run({{a, {1, 1}}}, {c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({3, 3}, {2}));

  auto d = Add(root, c, {39, 39});
  outputs.clear();
  TF_EXPECT_OK(session.Run({{a, {-10, 1}}}, {d}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({31, 42}, {2}));
}

TEST(ClientSessionTest, MultiThreadedWithDefaultThreadpool) {
  Scope root = Scope::NewRootScope();
  auto a = Add(root, {1, 2}, {3, 4});
  auto b = Mul(root, {1, 2}, {3, 4});
  ClientSession session(root, GetSessionOptions());
  {
    thread::ThreadPool thread_pool(Env::Default(), "pool", 2);
    thread_pool.Schedule([&session, a]() {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(session.Run({a}, &outputs));
      test::ExpectTensorEqual<int>(outputs[0],
                                   test::AsTensor<int>({4, 6}, {2}));
    });
    thread_pool.Schedule([&session, b]() {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(session.Run({b}, &outputs));
      test::ExpectTensorEqual<int>(outputs[0],
                                   test::AsTensor<int>({3, 8}, {2}));
    });
  }
  auto c = Sub(root, b, a);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run({c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({-1, 2}, {2}));
}

TEST(ClientSessionTest, MultiThreadedWithCustomThreadpool) {
  Scope root = Scope::NewRootScope();
  int num_threads = 3;
  auto a = Add(root, {1, 2}, {3, 4});
  auto b = Mul(root, {1, 2}, {3, 4});
  ClientSession session(root, GetSessionOptions());

  auto inter_op_threadpool =
      absl::make_unique<CustomThreadPoolImpl>(num_threads);
  ASSERT_EQ(inter_op_threadpool->GetNumScheduleCalled(), 0);

  auto intra_op_threadpool =
      absl::make_unique<CustomThreadPoolImpl>(num_threads);
  ASSERT_EQ(intra_op_threadpool->GetNumScheduleCalled(), 0);

  tensorflow::thread::ThreadPoolOptions threadPoolOptions;
  threadPoolOptions.inter_op_threadpool = inter_op_threadpool.get();
  threadPoolOptions.intra_op_threadpool = intra_op_threadpool.get();

  {
    thread::ThreadPool thread_pool(Env::Default(), "pool", 2);
    thread_pool.Schedule([&session, a]() {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(session.Run(RunOptions(), ClientSession::FeedType{}, {a}, {},
                               &outputs, nullptr, thread::ThreadPoolOptions()));
      test::ExpectTensorEqual<int>(outputs[0],
                                   test::AsTensor<int>({4, 6}, {2}));
    });
    thread_pool.Schedule([&session, b]() {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(session.Run(RunOptions(), ClientSession::FeedType{}, {b}, {},
                               &outputs, nullptr, thread::ThreadPoolOptions()));
      test::ExpectTensorEqual<int>(outputs[0],
                                   test::AsTensor<int>({3, 8}, {2}));
    });
  }
  auto c = Sub(root, b, a);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run(RunOptions(), ClientSession::FeedType{}, {c}, {},
                           &outputs, nullptr, thread::ThreadPoolOptions()));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({-1, 2}, {2}));
}

TEST(ClientSessionTest, CallableWithDefaultThreadPool) {
  Scope root = Scope::NewRootScope();
  auto a = Placeholder(root, DT_INT32);
  auto b = Placeholder(root, DT_INT32);
  auto c = Add(root, a, b);
  ClientSession session(root);
  std::vector<Tensor> outputs;

  CallableOptions options;
  options.add_feed(a.node()->name());
  options.add_feed(b.node()->name());
  options.add_fetch(c.node()->name());
  ClientSession::CallableHandle callable;
  TF_CHECK_OK(session.MakeCallable(options, &callable));
  TF_EXPECT_OK(session.RunCallable(
      callable, {test::AsTensor<int>({1}, {}), test::AsTensor<int>({41}, {})},
      &outputs, nullptr));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({42}, {}));
  TF_EXPECT_OK(session.ReleaseCallable(callable));
}

TEST(ClientSessionTest, CallableWithCustomThreadPool) {
  Scope root = Scope::NewRootScope();
  int num_threads = 3;

  TensorShape data_shape({1, 1});
  auto a = Placeholder(root, DT_INT32, Placeholder::Shape(data_shape));
  auto b = Placeholder(root, DT_INT32, Placeholder::Shape(data_shape));
  auto c = BatchMatMul(root, a, b);
  ClientSession session(root);
  std::vector<Tensor> outputs;

  auto inter_op_threadpool =
      absl::make_unique<CustomThreadPoolImpl>(num_threads);
  ASSERT_EQ(inter_op_threadpool->GetNumScheduleCalled(), 0);

  auto intra_op_threadpool =
      absl::make_unique<CustomThreadPoolImpl>(num_threads);
  ASSERT_EQ(intra_op_threadpool->GetNumScheduleCalled(), 0);

  tensorflow::thread::ThreadPoolOptions threadPoolOptions;
  threadPoolOptions.inter_op_threadpool = inter_op_threadpool.get();
  threadPoolOptions.intra_op_threadpool = intra_op_threadpool.get();

  CallableOptions options;
  options.add_feed(a.node()->name());
  options.add_feed(b.node()->name());
  options.add_fetch(c.node()->name());
  ClientSession::CallableHandle callable;
  TF_CHECK_OK(session.MakeCallable(options, &callable));

  // This is needed to have BatchMatMul computation be scheduled in the
  // intra_op_threadpool.
  absl::Barrier barrier(num_threads + 1);
  for (int i = 0; i < num_threads; i++) {
    intra_op_threadpool->Schedule([&barrier, num_threads]() {
      tensorflow::SetPerThreadMaxParallelism(num_threads - 1);
      barrier.Block();
    });
  }
  barrier.Block();

  TF_EXPECT_OK(session.RunCallable(
      callable,
      {test::AsTensor<int>({2}, {1, 1}), test::AsTensor<int>({10}, {1, 1})},
      &outputs, nullptr, threadPoolOptions));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({20}, {1, 1}));
  TF_EXPECT_OK(session.ReleaseCallable(callable));
  ASSERT_GT(inter_op_threadpool->GetNumScheduleCalled(), 0);
  ASSERT_GT(intra_op_threadpool->GetNumScheduleCalled(), 0);

  // Free intra_op_threadpool and wait for its threads to exit before freeing
  // other objects (e.g. barrier). This is needed to avoid data race.
  intra_op_threadpool.reset();
}

}  // namespace
}  // namespace tensorflow


// Helper function to run a session on a specific device and return the result.
tensorflow::Status RunSessionOnDevice(const std::string& device_name, 
                                      const Output& operation, 
                                      std::vector<Tensor>& outputs) {
  // Create a new root scope and place all operations on the specified device.
  Scope root = Scope::NewRootScope().WithDevice(device_name);

  // Create the session with the specified device
  ClientSession session(root);
  return session.Run({operation}, &outputs);
}

// Test for adding two constants across CPU and GPU.
TEST(CrossDeviceExecutionTest, AddOperationOnCpuAndGpu) {
  // Define the scope for the CPU computation.
  Scope cpu_scope = Scope::NewRootScope().WithDevice("/cpu:0");
  auto cpu_a = ops::Const(cpu_scope, 10);
  auto cpu_b = ops::Const(cpu_scope, 32);
  auto cpu_add = ops::Add(cpu_scope, cpu_a, cpu_b);

  // Define the scope for the GPU computation.
  Scope gpu_scope = Scope::NewRootScope().WithDevice("/gpu:0");
  auto gpu_a = ops::Const(gpu_scope, 10);
  auto gpu_b = ops::Const(gpu_scope, 32);
  auto gpu_add = ops::Add(gpu_scope, gpu_a, gpu_b);

  // Run the CPU computation.
  std::vector<Tensor> cpu_outputs;
  TF_EXPECT_OK(RunSessionOnDevice("/cpu:0", cpu_add, cpu_outputs));

  // Run the GPU computation.
  std::vector<Tensor> gpu_outputs;
  TF_EXPECT_OK(RunSessionOnDevice("/gpu:0", gpu_add, gpu_outputs));

  // Validate that the results are identical.
  test::ExpectTensorEqual<int>(cpu_outputs[0], gpu_outputs[0]);
}
// Test for multiplying two constants across CPU and GPU.
TEST(CrossDeviceExecutionTest, MulOperationOnCpuAndGpu) {
  // Define the scope for the CPU computation.
  Scope cpu_scope = Scope::NewRootScope().WithDevice("/cpu:0");
  auto cpu_a = ops::Const(cpu_scope, {2, 3});
  auto cpu_b = ops::Const(cpu_scope, {4, 5});
  auto cpu_mul = ops::Mul(cpu_scope, cpu_a, cpu_b);

  // Define the scope for the GPU computation.
  Scope gpu_scope = Scope::NewRootScope().WithDevice("/gpu:0");
  auto gpu_a = ops::Const(gpu_scope, {2, 3});
  auto gpu_b = ops::Const(gpu_scope, {4, 5});
  auto gpu_mul = ops::Mul(gpu_scope, gpu_a, gpu_b);

  // Run the CPU computation.
  std::vector<Tensor> cpu_outputs;
  TF_EXPECT_OK(RunSessionOnDevice("/cpu:0", cpu_mul, cpu_outputs));

  // Run the GPU computation.
  std::vector<Tensor> gpu_outputs;
  TF_EXPECT_OK(RunSessionOnDevice("/gpu:0", gpu_mul, gpu_outputs));

  // Validate that the results are identical.
  test::ExpectTensorEqual<int>(cpu_outputs[0], gpu_outputs[0]);
}

// Test for matrix multiplication across CPU and GPU.
TEST(CrossDeviceExecutionTest, MatMulOperationOnCpuAndGpu) {
  // Define the scope for the CPU computation.
  Scope cpu_scope = Scope::NewRootScope().WithDevice("/cpu:0");
  auto cpu_matrix_a = ops::Const(cpu_scope, {{1, 2}, {3, 4}});
  auto cpu_matrix_b = ops::Const(cpu_scope, {{5, 6}, {7, 8}});
  auto cpu_matmul = ops::MatMul(cpu_scope, cpu_matrix_a, cpu_matrix_b);

  // Define the scope for the GPU computation.
  Scope gpu_scope = Scope::NewRootScope().WithDevice("/gpu:0");
  auto gpu_matrix_a = ops::Const(gpu_scope, {{1, 2}, {3, 4}});
  auto gpu_matrix_b = ops::Const(gpu_scope, {{5, 6}, {7, 8}});
  auto gpu_matmul = ops::MatMul(gpu_scope, gpu_matrix_a, gpu_matrix_b);

  // Run the CPU computation.
  std::vector<Tensor> cpu_outputs;
  TF_EXPECT_OK(RunSessionOnDevice("/cpu:0", cpu_matmul, cpu_outputs));

  // Run the GPU computation.
  std::vector<Tensor> gpu_outputs;
  TF_EXPECT_OK(RunSessionOnDevice("/gpu:0", gpu_matmul, gpu_outputs));

  // Validate that the results are identical.
  test::ExpectTensorEqual<int>(cpu_outputs[0], gpu_outputs[0]);
}

// Test that operations are correctly placed on the device.
TEST(CrossDeviceExecutionTest, OperationPlacementTest) {
  Scope root = Scope::NewRootScope();

  // Place an operation on CPU.
  auto cpu_a = ops::Const(root.WithDevice("/cpu:0"), {1, 2});
  auto cpu_b = ops::Const(root.WithDevice("/cpu:0"), {3, 4});
  auto cpu_add = ops::Add(root.WithDevice("/cpu:0"), cpu_a, cpu_b);

  // Place an operation on GPU.
  auto gpu_a = ops::Const(root.WithDevice("/gpu:0"), {1, 2});
  auto gpu_b = ops::Const(root.WithDevice("/gpu:0"), {3, 4});
  auto gpu_add = ops::Add(root.WithDevice("/gpu:0"), gpu_a, gpu_b);

  // Verify the devices for each operation.
  ASSERT_EQ(cpu_add.node()->assigned_device_name(), "/cpu:0");
  ASSERT_EQ(gpu_add.node()->assigned_device_name(), "/gpu:0");

  // Run the CPU computation.
  std::vector<Tensor> cpu_outputs;
  TF_EXPECT_OK(RunSessionOnDevice("/cpu:0", cpu_add, cpu_outputs));

  // Run the GPU computation.
  std::vector<Tensor> gpu_outputs;
  TF_EXPECT_OK(RunSessionOnDevice("/gpu:0", gpu_add, gpu_outputs));

  // Validate that the results are identical.
  test::ExpectTensorEqual<int>(cpu_outputs[0], gpu_outputs[0]);
}

// Helper function to run a session and return the status.
tensorflow::Status RunSession(const Scope& root, const Output& operation,
                              std::vector<Tensor>& outputs) {
  ClientSession session(root);
  return session.Run({operation}, &outputs);
}

// Test case for invalid feeds (wrong data types or shapes).
TEST(ErrorHandlingTest, InvalidFeedTest) {
  Scope root = Scope::NewRootScope();

  // Placeholder expecting an int32 tensor.
  auto a = ops::Placeholder(root, DT_INT32);

  // Invalid feed: Feeding a float instead of an int.
  ClientSession session(root);
  std::vector<Tensor> outputs;
  Status status = session.Run({{a, Tensor(DT_FLOAT, TensorShape({}))}}, {}, &outputs);

  // Expect failure with an error status for invalid feed type.
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.error_message(), "expected int32"));

  // Invalid feed: Feeding a tensor of incompatible shape.
  Tensor wrong_shape_tensor(DT_INT32, TensorShape({2, 2}));
  status = session.Run({{a, wrong_shape_tensor}}, {}, &outputs);

  // Expect failure due to incompatible shape.
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.error_message(), "shapes must be equal"));
}

// Test case for invalid operations (e.g., divide by zero).
TEST(ErrorHandlingTest, InvalidOperationTest) {
  Scope root = Scope::NewRootScope();

  // Division operation: Attempt to divide by zero.
  auto x = ops::Const(root, 10);
  auto y = ops::Const(root, 0);  // Invalid divisor.
  auto div_op = ops::Div(root, x, y);

  std::vector<Tensor> outputs;
  Status status = RunSession(root, div_op, outputs);

  // Expect failure due to divide by zero.
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Attempt to divide by zero"));
}

// Test case for session timeout (simulating long computation).
TEST(ErrorHandlingTest, SessionTimeoutTest) {
  Scope root = Scope::NewRootScope();

  // Simple computation that should complete normally.
  auto a = ops::Const(root, 1);
  auto b = ops::Const(root, 2);
  auto add_op = ops::Add(root, a, b);

  ClientSession session(root);

  // Set a very short timeout for the session (to simulate timeout).
  SessionOptions options;
  options.config.mutable_experimental()->set_rpc_timeout_in_ms(1);  // 1ms timeout

  std::vector<Tensor> outputs;
  Status status = session.Run({add_op}, &outputs);

  // Expect timeout failure (since 1ms is too short for most computations).
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.error_message(), "timeout"));
}

// Test case for uninitialized tensors (attempting to use a tensor that was not initialized).
TEST(ErrorHandlingTest, UninitializedTensorTest) {
  Scope root = Scope::NewRootScope();

  // Placeholder that is never fed (remains uninitialized).
  auto uninitialized_placeholder = ops::Placeholder(root, DT_INT32);

  // Attempt to use the uninitialized placeholder in an operation.
  auto add_op = ops::Add(root, uninitialized_placeholder, {10});

  std::vector<Tensor> outputs;
  Status status = RunSession(root, add_op, outputs);

  // Expect failure due to uninitialized tensor.
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.error_message(), "uninitialized"));
}

