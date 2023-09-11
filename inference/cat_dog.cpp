#include <iostream>
#include <string>
#include <assert.h>
#include <png.h>
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

#include "onnxruntime_c_api.h"
#include "provider_options.h"

const OrtApi* g_ort = NULL;

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      std::cout << msg << std::endl;                         \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);


bool read_bin_to_float_vector(const std::string& file_path, std::vector<float>& input_data) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << file_path << std::endl;
        return false;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    input_data.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(input_data.data()), size);

    file.close();
    return true;
}

template <typename Iter>
void write_output_to_bin(const std::string& file_path, Iter begin, Iter end) {
  std::ofstream output_file(file_path, std::ios::out | std::ios::binary);
  if (!output_file.is_open()) {
      std::cerr << "Error: Cannot open the output file." << std::endl;
      return;
  }
  for (Iter it = begin; it != end; ++it) {
      float value = static_cast<float>(*it);
      output_file.write(reinterpret_cast<char*>(&value), sizeof(float));
  }
  output_file.close();
}

int run_inference(
    OrtSession* session,
    std::vector<float>& input_data, 
    const std::string& output_file_path,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    const char* input_name,
    const char* output_name)
{
    // Calculate and Check shape's size
    size_t input_data_size = 1;
    for (const auto& dim : input_shape) {
        input_data_size *= dim;
    }
    if (input_data.size() != input_data_size) {
        std::cerr << "Error: The input file does not match the expected input size.";
        return -1;
    }

    OrtMemoryInfo* memory_info;
    ORT_ABORT_ON_ERROR(
        g_ort->CreateCpuMemoryInfo(
            OrtArenaAllocator, 
            OrtMemTypeDefault, 
            &memory_info)
    );

    // Create Input Tensor
    OrtValue* input_tensor = NULL;
    ORT_ABORT_ON_ERROR(
        g_ort->CreateTensorWithDataAsOrtValue(
            memory_info, 
            input_data.data(), 
            input_data.size() * sizeof(float), 
            input_shape.data(), 
            input_shape.size(), 
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
            &input_tensor)
    );
    // Check Input Tensor
    assert(input_tensor != NULL);
    int is_tensor;
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);

    g_ort->ReleaseMemoryInfo(memory_info);

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};

    // Create Output Tensor
    size_t output_data_size = 1;
    for (const auto& dim : output_shape) {
        output_data_size *= dim;
    }
    std::vector<float> results_(output_data_size);

    OrtValue* output_tensor = NULL;
    ORT_ABORT_ON_ERROR(
       g_ort->CreateTensorWithDataAsOrtValue(
            memory_info, 
            results_.data(), 
            results_.size() * sizeof(float), 
            output_shape.data(), 
            output_shape.size(), 
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
            &output_tensor)
    );
    
    // RunInferencer
    ORT_ABORT_ON_ERROR(
        g_ort->Run(
            session, NULL, input_names, 
            (const OrtValue* const*)&input_tensor,
            1, output_names, 1, &output_tensor)
    );

    float* output_data;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_data));
    write_output_to_bin(output_file_path, output_data, output_data + output_data_size);

    // std::array<float, 2> output_data;
    // float* data_ptr = output_data.data();
    // ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&data_ptr));
    // write_output_to_bin(output_file_path, output_data.begin(), output_data.end());

    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);

    return 0;
}

void verify_input_output_count(OrtSession* session) {
    size_t count;
    ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
    assert(count == 1);
    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
    assert(count == 1);
}

int main(int argc, char* argv[]) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    ORTCHAR_T* model_path = argv[1];
    ORTCHAR_T* input_file = argv[2];
    ORTCHAR_T* output_file = argv[3];

    OrtEnv* env;
    ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "cat_dog", &env));
    OrtSessionOptions* session_options;
    ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

    OrtSession* session;
    ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));
    verify_input_output_count(session);

    std::vector<float> input_data;
    read_bin_to_float_vector(input_file, input_data);

    int ret = run_inference(
        session, input_data, output_file,
        {1, 3, 50, 50},
        {1, 2},
        "input",
        "output"
    );
    if (ret != 0) {
        std::cout << "inference failed." << std::endl;
    }
   
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseEnv(env);

    return 0;
}