/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OMStatus.h"

#include "core/OMUtils.h"
#include "core/OMKernelData.h"

#include "execute/OMKernelExecutionBuilder.h"
#include "execute/OMUtils.h"
#include "execute/OMRuntimeKernel.h"

#include "PALGRU.h"

using namespace onert_micro;
using namespace onert_micro::core;
using namespace onert_micro::execute;

namespace
{

constexpr uint32_t inputTensorIdx = 0;
constexpr uint32_t weightInputTensorIdx = 1;
constexpr uint32_t biasiInputTensorIdx = 2;
constexpr uint32_t weightHiddenTensorIdx = 3;
constexpr uint32_t biasiHiddenTensorIdx = 4;
constexpr uint32_t hiddenStateTensorIdx = 5;

constexpr uint32_t outputTensorIdx = 0;

} // namespace

OMStatus
onert_micro::execute::execute_kernel_CircleGRU(const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &runtime_context = execute_args.runtime_context;
  core::OMRuntimeStorage &runtime_storage = execute_args.runtime_storage;
  uint16_t op_index = execute_args.kernel_index;

  const circle::Tensor *input;
  const circle::Tensor *weight;
  const circle::Tensor *output;

  uint8_t *input_data;
  uint8_t *weight_data;
  uint8_t *bias_data;
  uint8_t *output_data;

  const circle::FullyConnectedOptions *options;
  // Read kernel
  {
    execute::OMRuntimeKernel runtime_kernel;
    runtime_kernel.readKernel(op_index, runtime_context);

    input = runtime_kernel.inputs[inputTensorIdx];
    weight = runtime_kernel.inputs[weightTensorIdx];
    output = runtime_kernel.outputs[outputTensorIdx];
    assert(input != nullptr);
    assert(weight != nullptr);
    // Bias can be nullptr
    assert(output != nullptr);

    runtime_kernel.getDataFromStorage(op_index, runtime_storage, runtime_context);

    input_data = runtime_kernel.inputs_data[inputTensorIdx];
    weight_data = runtime_kernel.inputs_data[weightTensorIdx];
    bias_data = runtime_kernel.inputs_data[biasTensorIdx];
    output_data = runtime_kernel.outputs_data[outputTensorIdx];
    assert(input_data != nullptr);
    assert(weight_data != nullptr);
    // Bias can be nullptr
    assert(output_data != nullptr);

    options = runtime_kernel.first_operator->builtin_options_as_FullyConnectedOptions();
  }

  OMStatus status;

  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
    {
      FullyConnectedParams params{};
      status = calculateActivationRange(options->fused_activation_function(),
                                        &params.float_activation_min, &params.float_activation_max);
      if (status != Ok)
        return status;

      status =
        pal::FullyConnected(params, core::utils::castInputData<float>(input_data),
                            OMRuntimeShape(weight), core::utils::castInputData<float>(weight_data),
                            core::utils::castInputData<float>(bias_data), OMRuntimeShape(output),
                            core::utils::castOutputData<float>(output_data));
    }
    break;
#endif // DIS_FLOAT
#ifndef DIS_QUANT
    case circle::TensorType_INT8:
    {
      FullyConnectedParams op_params{};

      calculateOpDataFullyConnected(input, weight, output, options->fused_activation_function(),
                                    op_params);

      status =
        pal::FullyConnected(op_params, core::utils::castInputData<int8_t>(input_data),
                            OMRuntimeShape(weight), core::utils::castInputData<int8_t>(weight_data),
                            core::utils::castInputData<int32_t>(bias_data), OMRuntimeShape(output),
                            core::utils::castOutputData<int8_t>(output_data));
    }
    break;
    case circle::TensorType_INT16:
    {
      FullyConnectedParams op_params{};

      calculateOpDataFullyConnected(input, weight, output, options->fused_activation_function(),
                                    op_params);

      status =
        pal::FullyConnected(op_params, core::utils::castInputData<int16_t>(input_data),
                            OMRuntimeShape(weight), core::utils::castInputData<int8_t>(weight_data),
                            core::utils::castInputData<int32_t>(bias_data), OMRuntimeShape(output),
                            core::utils::castOutputData<int16_t>(output_data));
    }
    break;
#endif // DIS_QUANT
    default:
    {
      status = UnsupportedType;
      assert(false && "Unsupported type.");
    }
  }

  return status;
}
