/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IMPORTER_EX_H__
#define __LUCI_IMPORTER_EX_H__

#include "luci/IR/Module.h"

#include <memory>
#include <string>

namespace luci
{

class ImporterEx final
{
public:
  ImporterEx() = default;

public:
  std::unique_ptr<Module> importVerifyModule(const std::string &input_path) const;
};

} // namespace luci

#endif // __LUCI_IMPORTER_EX_H__
