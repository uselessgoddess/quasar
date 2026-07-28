// Direct hipBLASLt go/no-go for the model's largest projection.
//
// The timed unit is the same 4096x640 @ 640x32768 forward GEMM and its two
// backward GEMMs measured by linear_backend_spike.rs.  Allocation, heuristic
// selection, initialization, validation and host transfers are outside timing.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace
{
constexpr int64_t rows = 4096;
constexpr int64_t input_features = 640;
constexpr int64_t output_features = 32768;
constexpr float loss_scale = 1024.0F;
constexpr size_t workspace_bytes = 256ULL * 1024ULL * 1024ULL;

void hip_check(hipError_t status, const char* expression, int line)
{
    if(status != hipSuccess)
    {
        std::cerr << "HIP error at line " << line << " for " << expression << ": "
                  << hipGetErrorString(status) << '\n';
        std::exit(1);
    }
}

void lt_check(hipblasStatus_t status, const char* expression, int line)
{
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        std::cerr << "hipBLASLt error at line " << line << " for " << expression
                  << ": status=" << static_cast<int>(status) << '\n';
        std::exit(1);
    }
}

#define HIP_CHECK(expression) hip_check((expression), #expression, __LINE__)
#define LT_CHECK(expression) lt_check((expression), #expression, __LINE__)

struct Problem
{
    hipblasLtMatrixLayout_t a{};
    hipblasLtMatrixLayout_t b{};
    hipblasLtMatrixLayout_t c{};
    hipblasLtMatrixLayout_t d{};
    hipblasLtMatmulDesc_t operation{};
    hipblasLtMatmulPreference_t preference{};
    hipblasLtMatmulHeuristicResult_t heuristic{};
    void* a_ptr{};
    void* b_ptr{};
    void* d_ptr{};
    uint64_t selected_workspace{};

    Problem(hipblasLtHandle_t handle,
            hipblasOperation_t trans_a,
            hipblasOperation_t trans_b,
            int64_t logical_m,
            int64_t logical_n,
            int64_t logical_k,
            int64_t a_rows,
            int64_t a_cols,
            int64_t b_rows,
            int64_t b_cols,
            void* a_data,
            void* b_data,
            void* d_data)
        : a_ptr(a_data), b_ptr(b_data), d_ptr(d_data)
    {
        LT_CHECK(hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, a_rows, a_cols, a_rows));
        LT_CHECK(hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, b_rows, b_cols, b_rows));
        LT_CHECK(
            hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, logical_m, logical_n, logical_m));
        LT_CHECK(
            hipblasLtMatrixLayoutCreate(&d, HIP_R_16F, logical_m, logical_n, logical_m));
        LT_CHECK(hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        LT_CHECK(hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
        LT_CHECK(hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
        LT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
        LT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
                                                       HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspace_bytes,
                                                       sizeof(workspace_bytes)));

        int returned = 0;
        LT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
            handle, operation, a, b, c, d, preference, 1, &heuristic, &returned));
        if(returned != 1)
        {
            std::cerr << "hipBLASLt found no solution for " << logical_m << 'x' << logical_n
                      << 'x' << logical_k << '\n';
            std::exit(1);
        }
        selected_workspace = heuristic.workspaceSize;
    }

    Problem(const Problem&) = delete;
    Problem& operator=(const Problem&) = delete;

    ~Problem()
    {
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatmulDescDestroy(operation);
        hipblasLtMatrixLayoutDestroy(d);
        hipblasLtMatrixLayoutDestroy(c);
        hipblasLtMatrixLayoutDestroy(b);
        hipblasLtMatrixLayoutDestroy(a);
    }

    void run(hipblasLtHandle_t handle, void* workspace, hipStream_t stream) const
    {
        const float alpha = 1.0F;
        const float beta = 0.0F;
        LT_CHECK(hipblasLtMatmul(handle,
                                operation,
                                &alpha,
                                a_ptr,
                                a,
                                b_ptr,
                                b,
                                &beta,
                                d_ptr,
                                c,
                                d_ptr,
                                d,
                                &heuristic.algo,
                                workspace,
                                selected_workspace,
                                stream));
    }
};

std::vector<__half> pattern(size_t length, int period, float center, float scale)
{
    std::vector<__half> values(length);
    for(size_t i = 0; i < length; ++i)
    {
        values[i] = __float2half((static_cast<float>(i % period) - center) / scale);
    }
    return values;
}

void* upload(const std::vector<__half>& values)
{
    void* result = nullptr;
    HIP_CHECK(hipMalloc(&result, values.size() * sizeof(__half)));
    HIP_CHECK(hipMemcpy(
        result, values.data(), values.size() * sizeof(__half), hipMemcpyHostToDevice));
    return result;
}

std::vector<__half> download(void* source, size_t length)
{
    std::vector<__half> values(length);
    HIP_CHECK(
        hipMemcpy(values.data(), source, length * sizeof(__half), hipMemcpyDeviceToHost));
    return values;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0
                                  : values[middle];
}

float cpu_forward(const std::vector<__half>& a,
                  const std::vector<__half>& b,
                  int64_t i,
                  int64_t j)
{
    double result = 0.0;
    for(int64_t k = 0; k < input_features; ++k)
    {
        result += static_cast<double>(__half2float(a[i + k * rows]))
                  * static_cast<double>(__half2float(b[k + j * input_features]));
    }
    return static_cast<float>(result);
}

float cpu_input_grad(const std::vector<__half>& scaled_output_grad,
                     const std::vector<__half>& b,
                     int64_t i,
                     int64_t k)
{
    double result = 0.0;
    for(int64_t j = 0; j < output_features; ++j)
    {
        result += static_cast<double>(__half2float(scaled_output_grad[i + j * rows]))
                  * static_cast<double>(__half2float(b[k + j * input_features]));
    }
    return static_cast<float>(result / loss_scale);
}

float cpu_weight_grad(const std::vector<__half>& a,
                      const std::vector<__half>& scaled_output_grad,
                      int64_t k,
                      int64_t j)
{
    double result = 0.0;
    for(int64_t i = 0; i < rows; ++i)
    {
        result += static_cast<double>(__half2float(a[i + k * rows]))
                  * static_cast<double>(__half2float(scaled_output_grad[i + j * rows]));
    }
    return static_cast<float>(result / loss_scale);
}

float max_sample_error(const std::vector<__half>& actual,
                       const std::vector<std::pair<size_t, float>>& expected,
                       float divisor)
{
    float result = 0.0F;
    for(const auto& [index, value] : expected)
    {
        result =
            std::max(result, std::abs(__half2float(actual[index]) / divisor - value));
    }
    return result;
}

size_t count_nonfinite(const std::vector<__half>& values)
{
    return static_cast<size_t>(
        std::count_if(values.begin(), values.end(), [](__half value) {
            return !std::isfinite(__half2float(value));
        }));
}
} // namespace

int main()
{
    const size_t a_elements = static_cast<size_t>(rows * input_features);
    const size_t b_elements = static_cast<size_t>(input_features * output_features);
    const size_t output_elements = static_cast<size_t>(rows * output_features);

    auto host_a = pattern(a_elements, 31, 15.0F, 64.0F);
    auto host_b = pattern(b_elements, 29, 14.0F, 128.0F);
    // This buffer is the scaled loss gradient. Gradients are unscaled for all
    // diagnostics below, matching the production dynamic scaler.
    auto host_output_grad = pattern(output_elements, 17, 8.0F, 64.0F);

    void* device_a = upload(host_a);
    void* device_b = upload(host_b);
    void* device_output_grad = upload(host_output_grad);
    void* device_output = nullptr;
    void* device_input_grad = nullptr;
    void* device_weight_grad = nullptr;
    void* workspace = nullptr;
    HIP_CHECK(hipMalloc(&device_output, output_elements * sizeof(__half)));
    HIP_CHECK(hipMalloc(&device_input_grad, a_elements * sizeof(__half)));
    HIP_CHECK(hipMalloc(&device_weight_grad, b_elements * sizeof(__half)));
    HIP_CHECK(hipMalloc(&workspace, workspace_bytes));

    hipblasLtHandle_t handle{};
    hipStream_t stream{};
    LT_CHECK(hipblasLtCreate(&handle));
    HIP_CHECK(hipStreamCreate(&stream));

    Problem forward(handle,
                    HIPBLAS_OP_N,
                    HIPBLAS_OP_N,
                    rows,
                    output_features,
                    input_features,
                    rows,
                    input_features,
                    input_features,
                    output_features,
                    device_a,
                    device_b,
                    device_output);
    Problem input_backward(handle,
                           HIPBLAS_OP_N,
                           HIPBLAS_OP_T,
                           rows,
                           input_features,
                           output_features,
                           rows,
                           output_features,
                           input_features,
                           output_features,
                           device_output_grad,
                           device_b,
                           device_input_grad);
    Problem weight_backward(handle,
                            HIPBLAS_OP_T,
                            HIPBLAS_OP_N,
                            input_features,
                            output_features,
                            rows,
                            rows,
                            input_features,
                            rows,
                            output_features,
                            device_a,
                            device_output_grad,
                            device_weight_grad);

    auto run_step = [&]() {
        forward.run(handle, workspace, stream);
        input_backward.run(handle, workspace, stream);
        weight_backward.run(handle, workspace, stream);
    };

    run_step();
    HIP_CHECK(hipStreamSynchronize(stream));
    std::cout << "warmup 1/1\n";

    hipEvent_t begin{};
    hipEvent_t end{};
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    std::vector<double> seconds;
    for(int sample = 0; sample < 9; ++sample)
    {
        HIP_CHECK(hipEventRecord(begin, stream));
        run_step();
        HIP_CHECK(hipEventRecord(end, stream));
        HIP_CHECK(hipEventSynchronize(end));
        float milliseconds = 0.0F;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, begin, end));
        seconds.push_back(static_cast<double>(milliseconds) / 1000.0);
        const double tflops =
            6.0 * rows * input_features * output_features / seconds.back() / 1.0e12;
        std::cout << "measured " << sample + 1 << "/"
                  << (sample < 3 ? 3 : 9) << " seconds=" << seconds.back()
                  << " throughput=" << tflops << " TFLOP/s\n";
        if(seconds.size() == 3)
        {
            const auto [minimum, maximum] =
                std::minmax_element(seconds.begin(), seconds.end());
            const double spread = 100.0 * (*maximum / *minimum - 1.0);
            if(spread <= 3.0)
            {
                break;
            }
            std::cout << "measurement window extended from 3 to 9 samples: initial "
                         "min/max spread "
                      << spread << "% exceeds 3%\n";
        }
    }

    auto host_output = download(device_output, output_elements);
    auto host_input_grad = download(device_input_grad, a_elements);
    auto host_weight_grad = download(device_weight_grad, b_elements);

    double activation_sum = 0.0;
    float activation_min = std::numeric_limits<float>::infinity();
    float activation_max = -std::numeric_limits<float>::infinity();
    for(__half value : host_output)
    {
        const float converted = __half2float(value);
        activation_sum += converted;
        activation_min = std::min(activation_min, converted);
        activation_max = std::max(activation_max, converted);
    }
    double grad_square_sum = 0.0;
    for(__half value : host_input_grad)
    {
        const double unscaled = static_cast<double>(__half2float(value)) / loss_scale;
        grad_square_sum += unscaled * unscaled;
    }

    std::vector<std::pair<size_t, float>> expected_output;
    std::vector<std::pair<size_t, float>> expected_input_grad;
    std::vector<std::pair<size_t, float>> expected_weight_grad;
    for(int64_t sample = 0; sample < 8; ++sample)
    {
        const int64_t i = (sample * 509) % rows;
        const int64_t j = (sample * 4093) % output_features;
        const int64_t k = (sample * 79) % input_features;
        expected_output.emplace_back(i + j * rows, cpu_forward(host_a, host_b, i, j));
        expected_input_grad.emplace_back(
            i + k * rows, cpu_input_grad(host_output_grad, host_b, i, k));
        expected_weight_grad.emplace_back(
            k + j * input_features, cpu_weight_grad(host_a, host_output_grad, k, j));
    }
    const float output_error = max_sample_error(host_output, expected_output, 1.0F);
    const float input_grad_error =
        max_sample_error(host_input_grad, expected_input_grad, loss_scale);
    const float weight_grad_error =
        max_sample_error(host_weight_grad, expected_weight_grad, loss_scale);
    const size_t nonfinite_count = count_nonfinite(host_output)
                                   + count_nonfinite(host_input_grad)
                                   + count_nonfinite(host_weight_grad);
    const double activation_mean =
        activation_sum / static_cast<double>(host_output.size());
    const double grad_norm = std::sqrt(grad_square_sum);
    std::cout << "precision activation_min=" << activation_min
              << " activation_max=" << activation_max
              << " activation_mean=" << activation_mean << " grad_norm=" << grad_norm
              << " loss_scale=" << loss_scale << " nonfinite_count=" << nonfinite_count
              << " output_max_error=" << output_error
              << " input_grad_max_error=" << input_grad_error
              << " weight_grad_max_error=" << weight_grad_error << '\n';

    const auto [minimum, maximum] = std::minmax_element(seconds.begin(), seconds.end());
    const double median_seconds = median(seconds);
    const double spread = 100.0 * (*maximum / *minimum - 1.0);
    const double tflops =
        6.0 * rows * input_features * output_features / median_seconds / 1.0e12;
    std::cout << "result backend=hipBLASLt dtype=F16 rows=" << rows << " shape="
              << input_features << 'x' << output_features << " samples=" << seconds.size()
              << " median_seconds=" << median_seconds << " min_seconds=" << *minimum
              << " max_seconds=" << *maximum << " spread=" << spread
              << "% throughput=" << tflops << " TFLOP/s\n";

    if(nonfinite_count != 0 || output_error > 1.0e-2F || input_grad_error > 1.0e-2F
       || weight_grad_error > 1.0e-2F)
    {
        std::cerr << "hipBLASLt precision gate failed\n";
        return 1;
    }

    HIP_CHECK(hipEventDestroy(end));
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipStreamDestroy(stream));
    LT_CHECK(hipblasLtDestroy(handle));
    HIP_CHECK(hipFree(workspace));
    HIP_CHECK(hipFree(device_weight_grad));
    HIP_CHECK(hipFree(device_input_grad));
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_grad));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_a));
    return 0;
}
