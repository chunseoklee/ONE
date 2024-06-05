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

#include "OMTrainingInterpreter.h"

using namespace onert_micro;

OMStatus OMTrainingInterpreter::importTrainModel(char *model_ptr, const OMConfig &config)
{
  assert(model_ptr != nullptr && "Model ptr shouldn't be nullptr");
  if (model_ptr == nullptr)
    return UnknownError;

  return _training_runtime_module.importTrainModel(model_ptr, config);
}

OMStatus OMTrainingInterpreter::trainSingleStep(const OMConfig &config)
{
  return _training_runtime_module.trainSingleStep(config);
}

OMStatus OMTrainingInterpreter::reset() { return _training_runtime_module.reset(); }

uint32_t OMTrainingInterpreter::getInputSizeAt(uint32_t position)
{
  return _training_runtime_module.getInputSizeAt(position);
}

uint32_t OMTrainingInterpreter::getOutputSizeAt(uint32_t position)
{
  return _training_runtime_module.getOutputSizeAt(position);
}
