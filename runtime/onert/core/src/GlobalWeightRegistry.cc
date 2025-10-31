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

/**
 * @file  GlobalWeightRegistry.cc
 * @brief This file implements global weight registry for external weight data management
 */

#include <unordered_map>
#include <string>
#include <mutex>

#include "GlobalWeightRegistry.h"

// Global registry for external weight data pointers
static std::unordered_map<std::string, const uint8_t*> g_weight_registry;
static std::mutex g_weight_registry_mutex;

/**
 * @brief Register weight data pointer for a specific key
 * 
 * @param key The key to identify the weight data
 * @param weight_data_ptr Pointer to the weight data
 * @return 0 on success, non-zero on error
 */
extern "C" int registerGlobalWeightData(const char* key, const uint8_t* weight_data_ptr)
{
  if (key == nullptr)
    return -1;
    
  std::lock_guard<std::mutex> lock(g_weight_registry_mutex);
  g_weight_registry[key] = weight_data_ptr;
  return 0;
}

/**
 * @brief Unregister weight data pointer for a specific key
 * 
 * @param key The key to identify the weight data
 * @return 0 on success, non-zero on error
 */
extern "C" int unregisterGlobalWeightData(const char* key)
{
  if (key == nullptr)
    return -1;
    
  std::lock_guard<std::mutex> lock(g_weight_registry_mutex);
  g_weight_registry.erase(key);
  return 0;
}

/**
 * @brief Get registered weight data pointer for a specific key
 * 
 * @param key The key to identify the weight data
 * @return Pointer to weight data, or nullptr if not found
 */
extern "C" const uint8_t* getGlobalWeightData(const char* key)
{
  if (key == nullptr)
    return nullptr;
    
  std::lock_guard<std::mutex> lock(g_weight_registry_mutex);
  auto it = g_weight_registry.find(key);
  if (it != g_weight_registry.end())
  {
    return it->second;
  }
  return nullptr;
}

/**
 * @brief Internal function for FullyConnectedLayer to access global weight data
 * 
 * This function provides direct access to the global weight registry for internal components
 * like FullyConnectedLayer without exposing the public API.
 * 
 * @param key The key to identify the weight data
 * @return Pointer to weight data, or nullptr if not found
 */
extern "C" const uint8_t* getInternalWeightData(const char* key)
{
  if (key == nullptr)
    return nullptr;
    
  std::lock_guard<std::mutex> lock(g_weight_registry_mutex);
  auto it = g_weight_registry.find(key);
  if (it != g_weight_registry.end())
  {
    return it->second;
  }
  return nullptr;
}
