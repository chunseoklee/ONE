/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "FullyConnectedLayer.h"

#include "GGMLHelper.h"

#include "../Tensor.h"
#include <cker/operation/FullyConnected.h>
#include <cker/TensorUtils.h>
#include <misc/polymorphic_downcast.h>
#include <fstream>
#include <unordered_map>
#include <mutex>

#include "../../../core/include/GlobalWeightRegistry.h"

namespace onert::backend::cpu::ops
{

// External C functions for global weight registry access
extern "C" {
  const uint8_t* getInternalWeightData(const char* key);
  int registerGlobalWeightData(const char* key, const uint8_t* weight_data_ptr);
  int unregisterGlobalWeightData(const char* key);
}

// Legacy functions for backward compatibility
void registerWeightData(const std::string& file_path, const uint8_t* weight_data_ptr)
{
  // This function is deprecated. Use nnfw_register_global_weight_data instead.
  // For backward compatibility, we still register the data using internal function.
  registerGlobalWeightData(file_path.c_str(), weight_data_ptr);
}

void unregisterWeightData(const std::string& file_path)
{
  // This function is deprecated. Use nnfw_unregister_global_weight_data instead.
  // For backward compatibility, we still unregister using internal function.
  unregisterGlobalWeightData(file_path.c_str());
}

const uint8_t* getRegisteredWeightData(const std::string& file_path)
{
  // Use the internal global weight registry access function
  return getInternalWeightData(file_path.c_str());
}

FullyConnectedLayer::FullyConnectedLayer()
  : _input(nullptr), _weights(nullptr), _bias(nullptr), _output(nullptr),
    _activation(ir::Activation::NONE), _temp_arena(new nnfw::cker::FCTempArena()),
    _external_context(nullptr), _is_hybrid(false), _is_shuffled16x1float32(false)
{
  // DO NOTHING
}

FullyConnectedLayer::~FullyConnectedLayer() = default;

void FullyConnectedLayer::fullyConnectedFloat32()
{
  nnfw::cker::FullyConnectedParams op_params;
  float output_activation_min = 0;
  float output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  op_params.activation = convertActivationType(_activation);
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  // TODO Set both cachables as false when training
  op_params.lhs_cacheable = _weights->is_constant();
  op_params.rhs_cacheable = _input->is_constant();

  nnfw::cker::FullyConnected(op_params, getShape(_input), getBuffer<float>(_input),
                             getShape(_weights), getBuffer<float>(_weights), getShape(_bias),
                             _bias ? getBuffer<float>(_bias) : nullptr, getShape(_output),
                             getBuffer<float>(_output));
}

// executionMutex is used to protect concurrent access of non-threadsafe resources
// like gemmlowp::GemmContext.
void FullyConnectedLayer::fullyConnectedQuant8()
{
  double real_multiplier = 0.0;
  int32_t output_multiplier = 0;
  int32_t output_shift = 0;
  int32_t output_activation_min = 0;
  int32_t output_activation_max = 0;
  GetQuantizedConvolutionMultiplier(_input, _weights, _bias, _output, &real_multiplier);
  QuantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
  CalculateActivationRangeQuantized(_activation, _output, &output_activation_min,
                                    &output_activation_max);

  nnfw::cker::FullyConnectedParams op_params;
  op_params.input_offset = -_input->data_zero_point();
  op_params.weights_offset = -_weights->data_zero_point();
  op_params.output_offset = _output->data_zero_point();
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  nnfw::cker::FullyConnected(op_params, getShape(_input), getBuffer<uint8_t>(_input),
                             getShape(_weights), getBuffer<uint8_t>(_weights), getShape(_bias),
                             _bias ? getBuffer<int32_t>(_bias) : nullptr, getShape(_output),
                             getBuffer<uint8_t>(_output));
}

void FullyConnectedLayer::fullyConnectedHybrid()
{
  nnfw::cker::FCTempArena &temp_arena = *_temp_arena;
  if (!temp_arena.prepared)
  {
    temp_arena.prepare(getShape(_input), getShape(_weights));
  }

  nnfw::cker::FullyConnectedParams op_params;
  op_params.activation = convertActivationType(_activation);
  op_params.weights_scale = _weights->data_scale();

#ifndef USE_RUY_GEMV
  nnfw::cker::FullyConnectedHybrid(
    op_params, getShape(_input), getBuffer<float>(_input), getShape(_weights),
    getBuffer<int8_t>(_weights), getShape(_bias), _bias ? getBuffer<float>(_bias) : nullptr,
    getShape(_output), getBuffer<float>(_output), temp_arena, _external_context->ruy_context());
#else
  nnfw::cker::FullyConnectedHybrid(
    op_params, getShape(_input), getBuffer<float>(_input), getShape(_weights),
    (_cached_weights) ? reinterpret_cast<const int8_t *>(_cached_weights)
                      : getBuffer<int8_t>(_weights),
    getShape(_bias), _bias ? getBuffer<float>(_bias) : nullptr, getShape(_output),
    getBuffer<float>(_output), temp_arena, _external_context->ruy_context());

  if (_cached_weights == nullptr || _is_weights_freed)
    return;

  // '_cached_weights is not nullptr and _is_weights_freed is false' means
  // this weight shape is satisfied with the ruy kernel's prepack cache's condition.
  // After entering here, it will not enter again except below the case - input is zero-vector

  // if input's elements are filled with zero, it by-passes(does not enter ruy-kernel path)
  // so that handle this case
  const int input_size = getShape(_input).FlatSize();
  if (nnfw::cker::IsZeroVector(getBuffer<float>(_input), input_size))
    return;

  auto weight_tensor = nnfw::misc::polymorphic_downcast<const Tensor *>(_weights);

  // This weight tensor could be other ops' const tensor.
  // Therefore, below reference should be checked like following
  auto tensor = const_cast<Tensor *>(weight_tensor);
  if (tensor->buffer() == nullptr) // ref is already 0?
  {
    _is_weights_freed = true;
    return;
  }

  tensor->decrease_ref();
  if (tensor->buffer() == nullptr) // ref == 0?
  {
#if defined(__ANDROID__) && (__ANDROID_API__ >= 26)
    // NOTE This line forces OS to release any unused memory immediately
    mallopt(M_PURGE, 0);
#endif
    _is_weights_freed = true;
  }
#endif
}

// FIXME: remove this after real w_ptr is provided
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#define NPUBIN_META_SIZE 4096

/**
 * Calculate extended metadata size from magiccode
 * 
 * @param magiccode The magic code from the TVN file
 * @return Extended metadata size in bytes
 */
inline uint64_t npubin_meta_extended_size(uint64_t magiccode) {
    // C++ implementation of: ((magiccode >> 8) & 0xFFULL) * NPUBIN_META_SIZE
    uint64_t num_extended = (magiccode >> 8) & 0xFF;
    return num_extended * NPUBIN_META_SIZE;
}

/**
 * Parse minimal metadata from TVN file
 * 
 * @param file_path Path to the TVN file
 * @param magiccode Output parameter for magiccode
 * @param program_size Output parameter for program size
 * @param extended_metasize Output parameter for extended metadata size
 * @return true if successful, false otherwise
 */
bool parse_minimal_metadata(const char* file_path, 
                           uint64_t* magiccode, 
                           uint64_t* program_size, 
                           uint32_t* extended_metasize) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Failed to open file %s\n", file_path);
        return false;
    }
    
    // Read base metadata (first 4096 bytes)
    char metadata[NPUBIN_META_SIZE];
    file.read(metadata, NPUBIN_META_SIZE);
    
    if (file.gcount() < NPUBIN_META_SIZE) {
        fprintf(stderr, "Error: File too small, expected at least %d bytes\n", NPUBIN_META_SIZE);
        file.close();
        return false;
    }
    
    // Parse magiccode (offset 0, 8 bytes, little-endian)
    *magiccode = static_cast<uint64_t>(static_cast<uint8_t>(metadata[0])) |
                 (static_cast<uint64_t>(static_cast<uint8_t>(metadata[1])) << 8) |
                 (static_cast<uint64_t>(static_cast<uint8_t>(metadata[2])) << 16) |
                 (static_cast<uint64_t>(static_cast<uint8_t>(metadata[3])) << 24) |
                 (static_cast<uint64_t>(static_cast<uint8_t>(metadata[4])) << 32) |
                 (static_cast<uint64_t>(static_cast<uint8_t>(metadata[5])) << 40) |
                 (static_cast<uint64_t>(static_cast<uint8_t>(metadata[6])) << 48) |
                 (static_cast<uint64_t>(static_cast<uint8_t>(metadata[7])) << 56);
    
    // Parse extended_metasize (offset 188, 4 bytes, little-endian)
    *extended_metasize = static_cast<uint32_t>(static_cast<uint8_t>(metadata[188])) |
                         (static_cast<uint32_t>(static_cast<uint8_t>(metadata[189])) << 8) |
                         (static_cast<uint32_t>(static_cast<uint8_t>(metadata[190])) << 16) |
                         (static_cast<uint32_t>(static_cast<uint8_t>(metadata[191])) << 24);
    
    // Parse program_size (offset 224, 8 bytes, little-endian)
    *program_size = static_cast<uint64_t>(static_cast<uint8_t>(metadata[224])) |
                    (static_cast<uint64_t>(static_cast<uint8_t>(metadata[225])) << 8) |
                    (static_cast<uint64_t>(static_cast<uint8_t>(metadata[226])) << 16) |
                    (static_cast<uint64_t>(static_cast<uint8_t>(metadata[227])) << 24) |
                    (static_cast<uint64_t>(static_cast<uint8_t>(metadata[228])) << 32) |
                    (static_cast<uint64_t>(static_cast<uint8_t>(metadata[229])) << 40) |
                    (static_cast<uint64_t>(static_cast<uint8_t>(metadata[230])) << 48) |
                    (static_cast<uint64_t>(static_cast<uint8_t>(metadata[231])) << 56);
    
    file.close();
    return true;
}

/**
 * Calculate weight offset from TVN file
 * 
 * @param tvn_path Path to the TVN file
 * @return Weight offset in bytes, or -1 if failed
 */
int64_t get_tvn_weight_offset(const char* tvn_path) {
    uint64_t magiccode, program_size;
    uint32_t extended_metasize;
    
    // Parse metadata
    if (!parse_minimal_metadata(tvn_path, &magiccode, &program_size, &extended_metasize)) {
        return -1;
    }
    
    // Calculate extended metadata size
    // Backward compatibility: extended_metasize가 0이면 legacy 방식 사용
    uint64_t extended_size;
    if (extended_metasize == 0) {
        extended_size = npubin_meta_extended_size(magiccode);
    } else {
        extended_size = extended_metasize;
    }
    
    // Calculate weight offset
    // NPUBIN_META_SIZE + extended_size + program_size
    int64_t weight_offset = static_cast<int64_t>(NPUBIN_META_SIZE) + 
                           static_cast<int64_t>(extended_size) + 
                           static_cast<int64_t>(program_size);
    
    printf("File: %s\n", tvn_path);
    printf("Magiccode: 0x%016lx\n", magiccode);
    printf("Program size: %ld bytes\n", program_size);
    printf("Extended metadata size: %ld bytes\n", extended_size);
    printf("Weight offset: %ld bytes (0x%lx)\n", weight_offset, weight_offset);
    
    return weight_offset;
}

/**
 * Get complete weight information from TVN file
 * 
 * @param tvn_path Path to the TVN file
 * @param weight_offset Output parameter for weight offset
 * @param weight_size Output parameter for weight size
 * @return true if successful, false otherwise
 */
bool get_tvn_weight_info(const char* tvn_path, int64_t* weight_offset, int64_t* weight_size) {
    uint64_t magiccode, program_size;
    uint32_t extended_metasize;
    
    // Parse metadata
    if (!parse_minimal_metadata(tvn_path, &magiccode, &program_size, &extended_metasize)) {
        return false;
    }
    
    // Calculate extended metadata size
    uint64_t extended_size;
    if (extended_metasize == 0) {
        extended_size = npubin_meta_extended_size(magiccode);
    } else {
        extended_size = extended_metasize;
    }
    
    // Calculate weight offset
    *weight_offset = static_cast<int64_t>(NPUBIN_META_SIZE) + 
                     static_cast<int64_t>(extended_size) + 
                     static_cast<int64_t>(program_size);
    
    // Get file size
    struct stat stat_buffer;
    if (stat(tvn_path, &stat_buffer) != 0) {
        return false;
    }
    
    // Calculate weight size (file size - weight offset)
    *weight_size = stat_buffer.st_size - *weight_offset;
    
    return true;
}


/**
 * @brief Execute fully connected layer with TRIX weight sharing quantization
 * 
 * This function handles TRIX quantized weights with weight sharing. It loads weight data
 * from external files and dispatches to the appropriate TRIX kernel based on the weight type.
 * 
 * Note: This implementation contains hardcoded file loading logic that should be replaced
 * with proper weight management in a production environment.
 */
void FullyConnectedLayer::fullyConnectediWeightShare()
{
  // Extract TRIX quantization parameters
  const auto &trix_quant = *_weights->get_info().typeInfo().trixQuantization();
 
  // Prepare operation parameters
  nnfw::cker::FullyConnectedParams op_params;
  op_params.activation = convertActivationType(_activation);
  op_params.weights_scale = _weights->data_scale();

  // Extract per-channel weight quantization parameters
  const auto *filter_per_channel_scales = _weights->data_scales().data();
  const auto *filter_per_channel_zp = _weights->data_zero_points().data();
   
  // Check if weight data is registered externally
  const uint8_t* weight_data_ptr = nullptr;
  const std::string& tvm_file_path = "model.tvn";  // Default path for backward compatibility
  
  // Try to get externally registered weight data first
  weight_data_ptr = getRegisteredWeightData(tvm_file_path);
  
  // If no external data is registered, fall back to file-based approach
  std::vector<uint8_t> file_data;
  if (weight_data_ptr == nullptr)
  {
    // TODO: FIXME - This is a temporary workaround for weight loading
    // In production, weight data should be provided through the proper tensor interface
    // rather than hardcoded file loading
    
    constexpr const char* TVN_FILE_PATH = "/mnt/ssd/dev/ONE/Product/x86_64-linux.debug/out/bin/model.tvn";
    long TVN_WEIGHT_OFFSET = get_tvn_weight_offset(TVN_FILE_PATH);

    // Load weight data from TVN file (temporary implementation)
    std::ifstream weight_file(TVN_FILE_PATH, std::ios::binary | std::ios::ate);
    if (!weight_file.is_open()) {
      throw std::runtime_error{"FullyConnected: Failed to open TVN weight file: " + std::string(TVN_FILE_PATH)};
    }
    
    const std::streamsize file_size = weight_file.tellg();
    weight_file.seekg(0, std::ios::beg);
    
    file_data.resize(file_size);
    weight_file.read(reinterpret_cast<char*>(file_data.data()), file_size);
    weight_file.close();
    
    weight_data_ptr = &file_data[TVN_WEIGHT_OFFSET];
  }
  
  // Dispatch to appropriate TRIX kernel based on weight quantization type
  switch (_weights->data_type()) {
    case OperandType::QUANT_TRIX_W4A8:
      // 4-bit weights, 8-bit activations
      nnfw::cker::FullyConnectedTRIXW4A8(op_params, 
                                         getShape(_input), 
                                         getBuffer<uint8_t>(_input),
                                         getShape(_weights), 
                                         weight_data_ptr,
                                         getShape(_bias), 
                                         _bias ? getBuffer<int32_t>(_bias) : nullptr,
                                         getShape(_output), 
                                         getBuffer<uint8_t>(_output), 
                                         trix_quant.in_ch_stride, 
                                         trix_quant.input_scale, 
                                         trix_quant.input_zp, 
                                         trix_quant.offset,
                                         filter_per_channel_scales, 
                                         filter_per_channel_zp);
      break;
      
    case OperandType::QUANT_TRIX_W8A8:
      // 8-bit weights, 8-bit activations
      nnfw::cker::FullyConnectedTRIXW8A8(op_params, 
                                         getShape(_input), 
                                         getBuffer<uint8_t>(_input),
                                         getShape(_weights), 
                                         weight_data_ptr,
                                         getShape(_bias), 
                                         _bias ? getBuffer<int32_t>(_bias) : nullptr,
                                         getShape(_output), 
                                         getBuffer<uint8_t>(_output), 
                                         trix_quant.in_ch_stride, 
                                         trix_quant.input_scale, 
                                         trix_quant.input_zp, 
                                         trix_quant.offset,
                                         filter_per_channel_scales, 
                                         filter_per_channel_zp);
      break;
      
    case OperandType::QUANT_TRIX_W8A16:
      // 8-bit weights, 16-bit activations
      nnfw::cker::FullyConnectedTRIXW8A16(op_params, 
                                          getShape(_input), 
                                          getBuffer<uint8_t>(_input),
                                          getShape(_weights), 
                                          weight_data_ptr,
                                          getShape(_bias), 
                                          _bias ? getBuffer<int32_t>(_bias) : nullptr,
                                          getShape(_output), 
                                          getBuffer<uint8_t>(_output), 
                                          trix_quant.in_ch_stride, 
                                          trix_quant.input_scale, 
                                          trix_quant.input_zp, 
                                          trix_quant.offset,
                                          filter_per_channel_scales, 
                                          filter_per_channel_zp);
      break;
      
    default:
      throw std::runtime_error{"FullyConnected: Unsupported TRIX weight type: " + 
                              std::to_string(static_cast<int>(_weights->data_type()))};
  }  
}

void FullyConnectedLayer::fullyConnectedSparseWeight()
{
  nnfw::cker::FullyConnectedParams op_params;
  op_params.activation = convertActivationType(_activation);

  const uint16_t *w1_segments = _weights->sparsity()->w1_segments();
  const uint16_t *w1_indices = _weights->sparsity()->w1_indices();

  auto block_size = _weights->sparsity()->block_size();
  if (block_size.size() == 0)
  {
    nnfw::cker::FullyConnectedSparseWeightRandom(
      op_params, getShape(_input), getBuffer<float>(_input), getShape(_weights),
      getBuffer<float>(_weights), getShape(_bias), _bias ? getBuffer<float>(_bias) : nullptr,
      getShape(_output), getBuffer<float>(_output), w1_segments, w1_indices);
  }
  else if (block_size.size() == 2 && block_size[0] == 16 && block_size[1] == 1)
  {
    nnfw::cker::FullyConnectedSparseWeight16x1(
      op_params, getShape(_input), getBuffer<float>(_input), getShape(_weights),
      getBuffer<float>(_weights), getShape(_bias), _bias ? getBuffer<float>(_bias) : nullptr,
      getShape(_output), getBuffer<float>(_output), w1_segments, w1_indices);
  }
  else
    throw std::runtime_error{"FullyConnected: unsupported sparsity"};
}

void FullyConnectedLayer::fullyConnectedGGMLWeight()
{
  if (_bias)
    throw std::runtime_error{"FullyConnected: GGML weights format does not support bias yet."};

  // convert tensor
  auto input = getGGMLTensor(_input);
  auto weights = getGGMLTensor(_weights);
  auto output = getGGMLTensor(_output);
  {
    output.op = GGML_OP_MUL_MAT;
    output.src[0] = &weights;
    output.src[1] = &input;
  }
  auto *nodes = &output;

  // create graph
  struct ggml_cgraph graph;
  {
    memset(&graph, 0, sizeof(graph));
    graph.n_nodes = 1;
    graph.nodes = &nodes;
  }

  // get cplan
  auto cplan = ggml_graph_plan(&graph, _external_context->maxNumThreads());
  std::vector<uint8_t> buf(cplan.work_size);
  cplan.work_data = buf.data();

  // compute
  ggml_graph_compute(&graph, &cplan);
}

void FullyConnectedLayer::fullyConnected16x1Float32()
{
#if defined(__aarch64__) && defined(USE_NEON)
  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  nnfw::cker::FullyConnectedParams op_params;
  op_params.activation = convertActivationType(_activation);

  nnfw::cker::FullyConnected16x1Float32(op_params, getShape(_input), getBuffer<float>(_input),
                                        getShape(_weights), getBuffer<float>(_weights),
                                        getShape(_bias), _bias ? getBuffer<float>(_bias) : nullptr,
                                        getShape(_output), getBuffer<float>(_output));
#else
  throw std::runtime_error{"FullyConnected: Shuffled16x1Float32 weights_format is not supported."};
#endif
}

void FullyConnectedLayer::configure(const IPortableTensor *input, const IPortableTensor *weights,
                                    const IPortableTensor *bias, ir::Activation activation,
                                    ir::FullyConnectedWeightsFormat weights_format,
                                    IPortableTensor *output,
                                    const std::shared_ptr<ExternalContext> &external_context)
{
  _input = input;
  _weights = weights;
  _bias = bias;
  _activation = activation;
  _output = output;
  _is_hybrid = input->data_type() == OperandType::FLOAT32 &&
               weights->data_type() == OperandType::QUANT_INT8_SYMM;
  _is_shuffled16x1float32 = weights_format == ir::FullyConnectedWeightsFormat::Shuffled16x1Float32;
#if !defined(__aarch64__) || !defined(USE_NEON)
  if (_is_shuffled16x1float32)
  {
    throw std::runtime_error{
      "FullyConnected: Shuffled16x1Float32 weights_format is not supported."};
  }
#endif
  _external_context = external_context;

  if (_weights->data_type() == OperandType::QUANT_GGML_Q4_0 ||
      _weights->data_type() == OperandType::QUANT_GGML_Q8_0)
    _external_context->initGgmlContext();
}

void FullyConnectedLayer::run()
{
  if (_is_hybrid)
  {
    fullyConnectedHybrid();
  }
  else if (_weights->sparsity())
  {
    fullyConnectedSparseWeight();
  }
  else if (_weights->data_type() == OperandType::QUANT_GGML_Q4_0 ||
           _weights->data_type() == OperandType::QUANT_GGML_Q8_0)
  {
    fullyConnectedGGMLWeight();
  }
  else if (_weights->data_type() == OperandType::QUANT_TRIX_W4A8 ||
           _weights->data_type() == OperandType::QUANT_TRIX_W8A8 ||
           _weights->data_type() == OperandType::QUANT_TRIX_W8A16 )
  {
    // For TRIX quantization, we should use weight sharing path
    fullyConnectediWeightShare();
  }
  else if (_input->data_type() == OperandType::FLOAT32)
  {
    _is_shuffled16x1float32 ? fullyConnected16x1Float32() : fullyConnectedFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    fullyConnectedQuant8();
  }
  else
  {
    throw std::runtime_error{"FullyConnected: unsupported data type"};
  }
}

void FullyConnectedLayer::prepare()
{
  if (_bias && _bias->is_constant())
  {
    const int bias_size = getShape(_bias).FlatSize();
    if (nnfw::cker::IsZeroVector(getBuffer<float>(_bias), bias_size))
    {
      _bias = nullptr;
    }
  }

#if (defined(__ARM_NEON__) || defined(__ARM_NEON)) && defined(USE_RUY_GEMV)
  // TODO This is workaround
  // The only fc hybrid will use ruy kernel
  if (_input->data_type() != OperandType::FLOAT32 ||
      _weights->data_type() != OperandType::QUANT_INT8_SYMM)
  {
    return;
  }

  // NOTE. The condition to enable caching on ruy kernel can be changed according to ruy's version

  // If input is dynamic, it changes total size of input
  // If weights is not constant, weights cannot be cached
  if (_input->is_dynamic() || !_weights->is_constant())
    return;

  const int rows = getShape(_weights).Dims(0);
  if (rows % 4 == 0)
  {
    // TODO If it's possible to extract precaching from ruy kernel,
    // place this instead of below code

    // buffer will be used by ruy kernel as a cache key
    _cached_weights = _weights->buffer();
  }
#endif
}

} // namespace onert::backend::cpu::ops
