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

#include "import/OMKernelConfigureBuilder.h"

#include "core/OMUtils.h"
#include "core/OMKernelData.h"

#include "execute/OMRuntimeKernel.h"

using namespace onert_micro;
using namespace onert_micro::core;

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
onert_micro::import::configure_kernel_CircleGRU(const OMConfigureArgs &config_args)
{
  OMRuntimeContext &runtime_context = config_args.runtime_context;
  uint16_t op_index = config_args.kernel_index;
  OMRuntimeStorage &runtime_storage = config_args.runtime_storage;

  execute::OMRuntimeKernel runtime_kernel;
  runtime_kernel.readKernel(op_index, runtime_context);

  const circle::Tensor *input = runtime_kernel.inputs[inputTensorIdx];
  const circle::Tensor *weight_input = runtime_kernel.inputs[weightInputTensorIdx];
  const circle::Tensor *bias_input = runtime_kernel.inputs[biasInputTensorIdx];
  const circle::Tensor *weight_hidden = runtime_kernel.inputs[weightHiddenTensorIdx];
  const circle::Tensor *bias_hidden = runtime_kernel.inputs[biasHiddenTensorIdx];
  const circle::Tensor *hidden_state = runtime_kernel.inputs[hiddenStateTensorIdx];
  const circle::Tensor *output = runtime_kernel.outputs[outputTensorIdx];

  assert(input != nullptr);
  assert(weight_input != nullptr);
  assert(weight_hidden != nullptr);
  // Bias can be nullptr
  assert(output != nullptr);

  OMStatus status = Ok;

  if ((input->type() != circle::TensorType_FLOAT32 ||
       weight->type() != circle::TensorType_FLOAT32))
  {
    return UnsupportedType;
  }

  core::OMRuntimeShape weight_input_shape(weight_input);
  core::OMRuntimeShape bias_input_shape(bias_input);
  core::OMRuntimeShape weight_hidden_shape(weight_hidden);
  core::OMRuntimeShape bias_hidden_shape(bias_hidden);
  core::OMRuntimeShape input_shape(input);
  core::OMRuntimeShape output_shape(output);

  status = utils::checkCondition(weight_input_shape.dimensionsCount() == 2);
  status = utils::checkCondition(weight_hidden_shape.dimensionsCount() == 2);
  if (status != Ok)
    return status;

  status = utils::checkCondition(bias_input == nullptr or weight_input_shape.dims(0) == bias_input_shape.flatSize());
  status = utils::checkCondition(bias_hidden == nullptr or weight_hidden_shape.dims(0) == bias_hidden_shape.flatSize());
  if (status != Ok)
    return status;


  return status;
}
