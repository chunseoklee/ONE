/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in riting, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_CPU_OPS_RANGELAYER_H__
#define __ONERT_BACKEND_CPU_OPS_RANGELAYER_H__

#include <backend/IPortableTensor.h>

#include <exec/IFunction.h>

namespace onert::backend::cpu::ops
{
class RangeLayer : public ::onert::exec::IFunction
{
public:
  RangeLayer();

  void configure(const IPortableTensor *start, const IPortableTensor *limit,
                 const IPortableTensor *delta, IPortableTensor *output);

  void run() override;

private:
  const IPortableTensor *_start;
  const IPortableTensor *_limit;
  const IPortableTensor *_delta;
  IPortableTensor *_output;
};

} // namespace onert::backend::cpu::ops

#endif // __ONERT_BACKEND_CPU_OPS_RANGELAYER_H__
