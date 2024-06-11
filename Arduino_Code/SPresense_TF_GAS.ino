//#include <TensorFlowLite.h>

#include "Gas_mat2.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"
//#include "tensorflow/lite/version.h"
const int kInputTensorSize = 110*1;
const int DIM1 = 110; // N
const int DIM2 = 1; // N
const int DIM3 = 1;
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int8_t* image_data = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 400 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace
void setup() {
  Serial.begin(115200);
  tflite::InitializeTarget();

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<3> micro_op_resolver;
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddFullyConnected();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  // Get information about the memory area to use for the model's input.


    //tflite::AllOpsResolver micro_op_resolver;

    // Build an interpreter to run the model with.
    // NOLINTNEXTLINE(runtime-global-variables)

    // Allocate memory from the tensor_arena for the model's tensors.

  /*for (size_t line = vert_top; line <= vert_bottom; line++) {
    for (size_t row = horz_left; row <= horz_right; row++, p++) {
      *image_data++ = tflite::FloatToQuantizedType<int8_t>(
          p[0] / 255.0f, tensor->params.scale, tensor->params.zero_point);
    }
    // move to next line
    p += ((image_width - 1) - horz_right) + horz_left;
  }*/
  /*if ((input->dims->size != 4) || (input->dims->data[0] != DIM1) ||
      (input->dims->data[1] != DIM2) ||
      (input->dims->data[2] != DIM3) ||
      (input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
    return;
  }  */


  
  // Reshape and partition the data
  
  //getInputInfo(input);
}

  void getInputInfo(TfLiteTensor* input){
  Serial.println("");
  Serial.println("Model input info");
  Serial.println("===============");
  Serial.print("Dimensions: ");
  Serial.println(input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(input->dims->data[1]);
  Serial.print("Dim 3 size: ");
  Serial.println(input->dims->data[2]);
  Serial.print("Dim 4 size: ");
  Serial.println(input->dims->data[3]);
  Serial.print("Input type: ");
  Serial.println(input->type);
  Serial.println("===============");
  Serial.println("");
  }



    // check if model loaded fine
    //if (!tf.isOk()) {
    //     Serial.print("ERROR: ");
    //    Serial.println(tf.getErrorMessage());
    //    
    //    while (true) delay(1000);
    //}

void loop() {
  if (Serial.available() >= kInputTensorSize * sizeof(float)) {
    // Read the incoming bytes and convert to float
    image_data = input->data.int8;
    for (int i = 0; i < kInputTensorSize; i++) {
      float incomingFloat;
      Serial.readBytes((char *)&incomingFloat, sizeof(float));
      *image_data++ = incomingFloat/input->params.scale + input->params.zero_point;; // Assuming the model takes float input
    }

    // Run inference
    if (interpreter->Invoke() == kTfLiteOk) {
      Serial.print((output->data.int8[0]- output->params.zero_point) * output->params.scale);
      Serial.print(", "); // Separate outputs by spaces
      Serial.print((output->data.int8[1]- output->params.zero_point) * output->params.scale);
      Serial.println(); // End the line to signal the end of response
    } else {
      Serial.println("Inference failed");
    }
  }
}
