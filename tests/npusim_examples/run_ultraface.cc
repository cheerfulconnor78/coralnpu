#include <stdint.h>
#include <stdio.h>
#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/opt/rvv_opt.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace {
using UltrafaceOpResolver = tflite::MicroMutableOpResolver<20>;
using coralnpu_v2::opt::litert_micro::Register_CONV_2D;

TfLiteStatus RegisterOps(UltrafaceOpResolver& op_resolver) {
  // TEST ISOLATION BOUNDARY:
  // Re-enable the custom optimized standard NPU convolution kernel
  op_resolver.AddConv2D(Register_CONV_2D());
  
  // Keep Depthwise Conv2D on the safe reference CPU path
  op_resolver.AddDepthwiseConv2D();
  
  op_resolver.AddReshape();
  op_resolver.AddConcatenation();
  op_resolver.AddMaxPool2D();
  op_resolver.AddAdd();
  op_resolver.AddSoftmax();
  op_resolver.AddPad();
  op_resolver.AddShape();
  op_resolver.AddStridedSlice();
  op_resolver.AddPack();

  // Core layout operators essential for any final layer array transformations
  op_resolver.AddTranspose();
  op_resolver.AddSqueeze();
  return kTfLiteOk;
}
} // namespace

extern "C" {
// Reserve memory for the model
uint8_t model_buffer[1500 * 1024] __attribute__((section(".data"), aligned(16)));

// Sized to a safe 4MB in external memory space (.extdata) to prevent internal DTCM truncation
uint8_t tensor_arena[4 * 1024 * 1024] __attribute__((section(".extdata"), aligned(16)));

// I/O buffers mapped to Python
uint8_t inference_input[240 * 320 * 3] __attribute__((section(".data"), aligned(16)));
int8_t inference_output_scores[4420 * 2] __attribute__((section(".data"), aligned(16)));
int8_t inference_output_boxes[4420 * 4] __attribute__((section(".data"), aligned(16)));
int8_t inference_status = -1;
} 

int main() {
  printf("Starting UltraFace Device Code...\n");
  const tflite::Model* model = tflite::GetModel(model_buffer);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("Model schema version %u is not supported by this version of TFMicro (%u).\n",
           (unsigned int)model->version(), (unsigned int)TFLITE_SCHEMA_VERSION);
    return -1;
  }

  UltrafaceOpResolver op_resolver;
  RegisterOps(op_resolver);

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, 4 * 1024 * 1024);

  printf("Allocating Tensors...\n");
  TfLiteStatus alloc_status = interpreter.AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    printf("Error: AllocateTensors failed with status code %d\n", (int)alloc_status);
    return -1;
  }

  // Debug Model Properties
  printf("Model Inputs: %d, Model Outputs: %d\n", (int)interpreter.inputs_size(), (int)interpreter.outputs_size());
  
  TfLiteTensor* input = interpreter.input(0);
  printf("Input Tensor Info: Type=%d, Bytes=%u, Dimensions:\n", (int)input->type, (unsigned int)input->bytes);
  for (int i = 0; i < input->dims->size; ++i) {
    printf("  dim[%d] = %d\n", i, (int)input->dims->data[i]);
  }

  // Safe Input Data Injection
  size_t copy_bytes = (input->bytes < sizeof(inference_input)) ? input->bytes : sizeof(inference_input);
  coralnpu_v2::opt::Memcpy(input->data.data, inference_input, copy_bytes);

  printf("Invoking Interpreter...\n");
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    printf("Error: Invoke failed with status code %d\n", (int)invoke_status);
    return -1;
  }

  // Dynamically Route Flipped Output Layer Pointers
  TfLiteTensor* out_0 = interpreter.output(0);
  TfLiteTensor* out_1 = interpreter.output(1);
  printf("Output 0 Bytes: %u, Output 1 Bytes: %u\n", (unsigned int)out_0->bytes, (unsigned int)out_1->bytes);

  TfLiteTensor* boxes_tensor = nullptr;
  TfLiteTensor* scores_tensor = nullptr;

  if (out_0->bytes == 4420 * 4) {
    boxes_tensor = out_0;
    scores_tensor = out_1;
  } else {
    boxes_tensor = out_1;
    scores_tensor = out_0;
  }

  // Safely copy results to mapped buffers
  coralnpu_v2::opt::Memcpy(inference_output_scores, scores_tensor->data.data, 4420 * 2);
  coralnpu_v2::opt::Memcpy(inference_output_boxes, boxes_tensor->data.data, 4420 * 4);

  printf("Inference Complete!\n");
  inference_status = 0;
  return 0;
}
