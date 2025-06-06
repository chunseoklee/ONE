/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_CPU_OPS_PADLAYER_H__
#define __ONERT_BACKEND_CPU_OPS_PADLAYER_H__

#include <backend/IPortableTensor.h>
#include "OperationUtils.h"

#include <exec/IFunction.h>

namespace onert::backend::cpu::ops
{

// Note, this is pad with mode=`CONSTANT`: it doesn't support `REFLECT` and
// `SYMMETRIC`
class PadLayer : public ::onert::exec::IFunction
{
public:
  PadLayer();

public:
  template <typename T> void padImpl(const T *constant_value_data);

  void configure(const IPortableTensor *input, const IPortableTensor *pad,
                 const IPortableTensor *value, IPortableTensor *output);

  void run() override;

protected:
  const IPortableTensor *_input;
  const IPortableTensor *_pad;
  const IPortableTensor *_value;
  IPortableTensor *_output;
  ConstDataPtr _constantValueData;
};

} // namespace onert::backend::cpu::ops

#endif // __ONERT_BACKEND_CPU_OPS_PADLAYER_H__
