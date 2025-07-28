#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <unistd.h>

void run_inference(char* onnx_filename, int num_runs)
{
    // Create runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "inference");
    Ort::SessionOptions session_options;
    Ort::Session session(env, onnx_filename, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input info
    auto input_name = session.GetInputNameAllocated(0, allocator);
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    size_t input_tensor_size = 1 * 3 * 224 * 224;
    std::vector<float> input_tensor_values(input_tensor_size, 1.0f);

    // Create input tensor. Running on CPU, can change based on hardware!
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor_values.data(), input_tensor_size,
        input_shape.data(), input_shape.size());

    // Get output name
    auto output_name = session.GetOutputNameAllocated(0, allocator);

    // Run inference
    double latency = 0;
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    // Warmup to eliminate memory optimization overhead behind the scenes
    for (int i=0; i<10; i++)
    {
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    }

    // Inference runs
    for (int i=0; i<num_runs; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                        input_names,
                                        &input_tensor, 1,
                                        output_names, 1);
        auto end = std::chrono::high_resolution_clock::now();
        latency += std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::cout << "\nInference for " << onnx_filename << " completed in " << latency/1000 << " ms\n";
}

int main() {
    run_inference((char*) "resnet-18.onnx", 1000);
    run_inference((char*) "resnet-18_quantized.onnx", 1000);
    std::cout << "\n";
    return 0;
}
