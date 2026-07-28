// Native tied-head GEMM seam for the ROCm backend.
//
// Burn owns every allocation. This wrapper only borrows the device pointers,
// selects a hipBLASLt algorithm once per shape, and executes forward or the two
// backward GEMMs on one private stream. Device synchronization is deliberate:
// CubeCL does not expose its HIP stream, so this is the smallest correctness
// boundary available. The full optimizer-step benchmark decides whether the
// synchronization cost is acceptable.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace
{
constexpr size_t workspace_bytes = 256ULL * 1024ULL * 1024ULL;
thread_local std::string last_error;

void hip_check(hipError_t status, const char* expression, int line)
{
    if(status != hipSuccess)
    {
        std::ostringstream message;
        message << "HIP error at line " << line << " for " << expression << ": "
                << hipGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

void lt_check(hipblasStatus_t status, const char* expression, int line)
{
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        std::ostringstream message;
        message << "hipBLASLt error at line " << line << " for " << expression
                << ": status=" << static_cast<int>(status);
        throw std::runtime_error(message.str());
    }
}

#define HIP_CHECK(expression) hip_check((expression), #expression, __LINE__)
#define LT_CHECK(expression) lt_check((expression), #expression, __LINE__)

using ProblemKey =
    std::tuple<int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

class Problem
{
  public:
    Problem(hipblasLtHandle_t handle,
            hipblasOperation_t trans_a,
            hipblasOperation_t trans_b,
            int64_t logical_m,
            int64_t logical_n,
            int64_t a_rows,
            int64_t a_cols,
            int64_t b_rows,
            int64_t b_cols)
    {
        LT_CHECK(hipblasLtMatrixLayoutCreate(&a_, HIP_R_16F, a_rows, a_cols, a_rows));
        LT_CHECK(hipblasLtMatrixLayoutCreate(&b_, HIP_R_16F, b_rows, b_cols, b_rows));
        LT_CHECK(
            hipblasLtMatrixLayoutCreate(&c_, HIP_R_16F, logical_m, logical_n, logical_m));
        LT_CHECK(
            hipblasLtMatrixLayoutCreate(&d_, HIP_R_16F, logical_m, logical_n, logical_m));
        LT_CHECK(hipblasLtMatmulDescCreate(&operation_, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        LT_CHECK(hipblasLtMatmulDescSetAttribute(
            operation_, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
        LT_CHECK(hipblasLtMatmulDescSetAttribute(
            operation_, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
        LT_CHECK(hipblasLtMatmulPreferenceCreate(&preference_));
        LT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference_,
                                                       HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspace_bytes,
                                                       sizeof(workspace_bytes)));

        int returned = 0;
        LT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
            handle, operation_, a_, b_, c_, d_, preference_, 1, &heuristic_, &returned));
        if(returned != 1)
        {
            throw std::runtime_error("hipBLASLt found no algorithm for tied-head GEMM");
        }
    }

    Problem(const Problem&) = delete;
    Problem& operator=(const Problem&) = delete;

    ~Problem()
    {
        hipblasLtMatmulPreferenceDestroy(preference_);
        hipblasLtMatmulDescDestroy(operation_);
        hipblasLtMatrixLayoutDestroy(d_);
        hipblasLtMatrixLayoutDestroy(c_);
        hipblasLtMatrixLayoutDestroy(b_);
        hipblasLtMatrixLayoutDestroy(a_);
    }

    void run(hipblasLtHandle_t handle,
             const void* a,
             const void* b,
             void* output,
             void* workspace,
             hipStream_t stream) const
    {
        const float alpha = 1.0F;
        const float beta = 0.0F;
        LT_CHECK(hipblasLtMatmul(handle,
                                operation_,
                                &alpha,
                                a,
                                a_,
                                b,
                                b_,
                                &beta,
                                output,
                                c_,
                                output,
                                d_,
                                &heuristic_.algo,
                                workspace,
                                heuristic_.workspaceSize,
                                stream));
    }

  private:
    hipblasLtMatrixLayout_t a_{};
    hipblasLtMatrixLayout_t b_{};
    hipblasLtMatrixLayout_t c_{};
    hipblasLtMatrixLayout_t d_{};
    hipblasLtMatmulDesc_t operation_{};
    hipblasLtMatmulPreference_t preference_{};
    hipblasLtMatmulHeuristicResult_t heuristic_{};
};

class Engine
{
  public:
    Engine()
    {
        LT_CHECK(hipblasLtCreate(&handle_));
        HIP_CHECK(hipStreamCreate(&stream_));
        HIP_CHECK(hipMalloc(&workspace_, workspace_bytes));
    }

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    ~Engine()
    {
        const hipError_t free_status = hipFree(workspace_);
        const hipError_t stream_status = hipStreamDestroy(stream_);
        static_cast<void>(free_status);
        static_cast<void>(stream_status);
        hipblasLtDestroy(handle_);
    }

    void forward(const void* hidden,
                 const void* weight,
                 void* output,
                 int64_t rows,
                 int64_t hidden_size,
                 int64_t vocab_size)
    {
        const std::scoped_lock lock(mutex_);
        HIP_CHECK(hipDeviceSynchronize());

        // Row-major [rows, hidden] and [vocab, hidden] buffers are viewed as
        // column-major [hidden, rows] and [hidden, vocab]. Compute
        // output^T = weight * hidden^T directly into row-major output.
        problem(HIPBLAS_OP_T,
                HIPBLAS_OP_N,
                vocab_size,
                rows,
                hidden_size,
                hidden_size,
                vocab_size,
                hidden_size,
                rows)
            .run(handle_, weight, hidden, output, workspace_, stream_);
        HIP_CHECK(hipStreamSynchronize(stream_));
    }

    void backward(const void* hidden,
                  const void* weight,
                  const void* output_grad,
                  void* hidden_grad,
                  void* weight_grad,
                  int64_t rows,
                  int64_t hidden_size,
                  int64_t vocab_size)
    {
        const std::scoped_lock lock(mutex_);
        HIP_CHECK(hipDeviceSynchronize());

        // dHidden^T = weight^T * dOutput^T.
        problem(HIPBLAS_OP_N,
                HIPBLAS_OP_N,
                hidden_size,
                rows,
                vocab_size,
                hidden_size,
                vocab_size,
                vocab_size,
                rows)
            .run(handle_, weight, output_grad, hidden_grad, workspace_, stream_);
        // dWeight^T = hidden^T * dOutput.
        problem(HIPBLAS_OP_N,
                HIPBLAS_OP_T,
                hidden_size,
                vocab_size,
                rows,
                hidden_size,
                rows,
                vocab_size,
                rows)
            .run(handle_, hidden, output_grad, weight_grad, workspace_, stream_);
        HIP_CHECK(hipStreamSynchronize(stream_));
    }

  private:
    Problem& problem(hipblasOperation_t trans_a,
                     hipblasOperation_t trans_b,
                     int64_t logical_m,
                     int64_t logical_n,
                     int64_t logical_k,
                     int64_t a_rows,
                     int64_t a_cols,
                     int64_t b_rows,
                     int64_t b_cols)
    {
        const ProblemKey key{static_cast<int>(trans_a),
                             static_cast<int>(trans_b),
                             logical_m,
                             logical_n,
                             logical_k,
                             a_rows,
                             a_cols,
                             b_rows,
                             b_cols};
        auto [entry, inserted] = problems_.try_emplace(key);
        if(inserted)
        {
            entry->second = std::make_unique<Problem>(handle_,
                                                      trans_a,
                                                      trans_b,
                                                      logical_m,
                                                      logical_n,
                                                      a_rows,
                                                      a_cols,
                                                      b_rows,
                                                      b_cols);
        }
        return *entry->second;
    }

    hipblasLtHandle_t handle_{};
    hipStream_t stream_{};
    void* workspace_{};
    std::map<ProblemKey, std::unique_ptr<Problem>> problems_;
    std::mutex mutex_;
};

Engine& engine()
{
    static Engine instance;
    return instance;
}

template <typename Function> int guarded(Function&& function)
{
    try
    {
        function();
        last_error.clear();
        return 0;
    }
    catch(const std::exception& error)
    {
        last_error = error.what();
        return 1;
    }
}
} // namespace

extern "C" int quasar_hipblaslt_tied_head_forward(const void* hidden,
                                                   const void* weight,
                                                   void* output,
                                                   int64_t rows,
                                                   int64_t hidden_size,
                                                   int64_t vocab_size)
{
    return guarded(
        [&] { engine().forward(hidden, weight, output, rows, hidden_size, vocab_size); });
}

extern "C" int quasar_hipblaslt_tied_head_backward(const void* hidden,
                                                    const void* weight,
                                                    const void* output_grad,
                                                    void* hidden_grad,
                                                    void* weight_grad,
                                                    int64_t rows,
                                                    int64_t hidden_size,
                                                    int64_t vocab_size)
{
    return guarded([&] {
        engine().backward(hidden,
                          weight,
                          output_grad,
                          hidden_grad,
                          weight_grad,
                          rows,
                          hidden_size,
                          vocab_size);
    });
}

extern "C" const char* quasar_hipblaslt_last_error()
{
    return last_error.c_str();
}
