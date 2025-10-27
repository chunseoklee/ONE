/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_FULLY_CONNECTED_H__
#define __NNFW_CKER_FULLY_CONNECTED_H__

#include <ruy/context.h>
#include "cker/operation/FullyConnectedDense16x1.h"
#include "cker/operation/FullyConnectedSparse16x1.h"
#include "cker/operation/optimized/Gemm.h"
#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include "cker/TensorUtils.h"
#include "cker/neon/neon_check.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
namespace nnfw
{
namespace cker
{

class FCTempArena
{
public:
  FCTempArena(void) : prepared(false), input_quantized(), scaling_factors(), accum_scratch()
  {
    // DO NOTHING
  }

  void prepare(const Shape &input_shape, const Shape &weights_shape)
  {
    auto input_size = input_shape.FlatSize();
    input_quantized.resize(input_size);

    assert(weights_shape.DimensionsCount() == 2);
    int batch_size = input_size / weights_shape.Dims(1);
    scaling_factors.resize(batch_size);
    prepared = true;
  }

public:
  bool prepared;
  std::vector<int8_t> input_quantized;
  std::vector<float> scaling_factors;
  std::vector<int32_t> accum_scratch;
};

#if defined(CKER_X86_PLATFORM)

// From tensorflow/tensorflow/lite/kernels/internal/optimized/optimized_ops.h
inline void FullyConnected(const FullyConnectedParams &params, const Shape &input_shape,
                           const float *input_data, const Shape &weights_shape,
                           const float *weights_data, const Shape &,
                           const float *optional_bias_data, const Shape &output_shape,
                           float *output_data)
{
  const int dims_count = weights_shape.DimensionsCount();
  const int input_rows = weights_shape.Dims(dims_count - 1);
  MatrixParams<float> rhs_params;
  rhs_params.order = Order::kColMajor;
  rhs_params.rows = input_rows;
  rhs_params.cols = input_shape.FlatSize() / input_rows;
  rhs_params.cache_policy = optimized::DefaultCachePolicy(params.rhs_cacheable);

  MatrixParams<float> lhs_params;
  lhs_params.order = Order::kRowMajor;
  lhs_params.cols = weights_shape.Dims(dims_count - 1);
  lhs_params.rows = FlatSizeSkipDim(weights_shape, dims_count - 1);
  lhs_params.cache_policy = optimized::DefaultCachePolicy(params.lhs_cacheable);
  MatrixParams<float> dst_params;
  dst_params.order = Order::kColMajor;
  dst_params.rows = output_shape.Dims(output_shape.DimensionsCount() - 1);
  dst_params.cols = FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  GemmParams<float, float> gemm_params;
  gemm_params.bias = optional_bias_data;
  gemm_params.clamp_min = params.float_activation_min;
  gemm_params.clamp_max = params.float_activation_max;
  optimized::Gemm(lhs_params, weights_data, rhs_params, input_data, dst_params, output_data,
                  gemm_params);
}

#else // CKER_X86_PLATFORM

inline void FullyConnected(const FullyConnectedParams &params, const Shape &input_shape,
                           const float *input_data, const Shape &weights_shape,
                           const float *weights_data, const Shape &, const float *bias_data,
                           const Shape &, float *output_data)
{
  int total_input_size = input_shape.FlatSize();
  int input_size = weights_shape.Dims(1);
  const int batch_size = total_input_size / input_size;
  const int num_units = weights_shape.Dims(0);

  // Output = bias if bias tensor exists.
  if (bias_data)
  {
    VectorBatchVectorAssign(bias_data, num_units, batch_size, output_data);
  }
  else
  {
    ZeroVector(output_data, batch_size * num_units);
  }

  // Compute output += weight * input
  MatrixBatchVectorMultiplyAccumulate(weights_data, num_units, input_size, input_data, batch_size,
                                      output_data, /*result_stride=*/1);

  if (params.activation != FusedActivationFunctionType::kNone)
  {
    // Apply activation function
    ApplyActivationToVector(output_data, batch_size * num_units, params.activation, output_data);
  }
}

#endif // CKER_X86_PLATFORM

inline void FullyConnected(const FullyConnectedParams &params,
                           [[maybe_unused]] const Shape &input_shape, const uint8_t *input_data,
                           const Shape &filter_shape, const uint8_t *filter_data,
                           [[maybe_unused]] const Shape &bias_shape, const int32_t *bias_data,
                           const Shape &output_shape, uint8_t *output_data)
{
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  assert(filter_shape.DimensionsCount() >= 2);
  assert(output_shape.DimensionsCount() >= 1);

  assert(output_activation_min <= output_activation_max);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth =
    MatchingDim(filter_shape, filter_dim_count - 2, output_shape, output_dim_count - 1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b)
  {
    for (int out_c = 0; out_c < output_depth; ++out_c)
    {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d)
      {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data)
      {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<uint8_t>(acc);
    }
  }
}

inline void FullyConnectedHybrid(const FullyConnectedParams &params, const Shape &input_shape,
                                 const float *input_data, const Shape &filter_shape,
                                 const int8_t *filter_data, const Shape &, const float *bias_data,
                                 [[maybe_unused]] const Shape &output_shape, float *output_data,
                                 FCTempArena &temp_arena,
                                 [[maybe_unused]] ruy::Context *ruy_context)
{
  int total_input_size = input_shape.FlatSize();
  const int input_size = filter_shape.Dims(1);
  const int batch_size = total_input_size / input_size;
  const int num_units = filter_shape.Dims(0);

  // Output = bias if bias tensor exists.
  if (bias_data)
  {
    VectorBatchVectorAssign(bias_data, num_units, batch_size, output_data);
  }
  else
  {
    ZeroVector(output_data, batch_size * num_units);
  }

  // Save matrix multiplication computation for all zero input.
  if (IsZeroVector(input_data, total_input_size))
  {
    ApplyActivationToVector(output_data, batch_size * num_units, params.activation, output_data);
    return;
  }

  // Quantize input from float to uint8 + quantization params (scaling factor).
  float unused_min, unused_max;
  float *scaling_factors_ptr = temp_arena.scaling_factors.data();
  int8_t *quant_data = temp_arena.input_quantized.data();

  // Quantize each batch independently.
  for (int b = 0; b < batch_size; ++b)
  {
    const int offset = b * input_size;
    SymmetricQuantizeFloats(input_data + offset, input_size, quant_data + offset, &unused_min,
                            &unused_max, &scaling_factors_ptr[b]);
    // Incorporate scaling of the filter.
    scaling_factors_ptr[b] *= params.weights_scale;
  }

// Compute output += weight * quantized_input
#ifdef USE_RUY_GEMV
  auto output_size = output_shape.FlatSize();
  temp_arena.accum_scratch.resize(output_size);
  int32_t *scratch = temp_arena.accum_scratch.data();
  MatrixBatchVectorMultiplyAccumulate(filter_data, num_units, input_size, quant_data,
                                      scaling_factors_ptr, batch_size, scratch, output_data,
                                      /*result_stride=*/1, ruy_context);
#else
  MatrixBatchVectorMultiplyAccumulate(filter_data, num_units, input_size, quant_data,
                                      scaling_factors_ptr, batch_size, output_data,
                                      /*result_stride=*/1);
#endif

  // Apply activation function to floats.
  if (params.activation != FusedActivationFunctionType::kNone)
  {
    // Apply activation function
    ApplyActivationToVector(output_data, batch_size * num_units, params.activation, output_data);
  }
  return;
}

/**
 * @brief Reference implementation for quantizing float32 values to uint8_t
 * 
 * This function quantizes floating-point values to 8-bit unsigned integers using
 * the provided scale and zero point parameters.
 * 
 * @param x Input float32 array
 * @param input_quantized Output uint8_t array for quantized values
 * @param input_scale Scale factor for quantization
 * @param input_zp Zero point for quantization
 * @param k Number of elements to quantize
 */
void quantize_q8a_tr_reference(const float *x, uint8_t *input_quantized, float input_scale, int32_t input_zp, int64_t k) {
    // Calculate inverse scale to avoid division in the loop
    const float inverse_scale = (input_scale != 0.0f) ? (1.0f / input_scale) : 0.0f;
    
    // Quantize each element: quantized_value = round(float_value / scale) + zero_point
    for (int64_t i = 0; i < k; i++) {
        const float scaled_value = x[i] * inverse_scale + input_zp;
        const int rounded_value = static_cast<int>(std::round(scaled_value));
        
        // Clamp to uint8_t range [0, 255]
        input_quantized[i] = static_cast<uint8_t>(std::max(0, std::min(255, rounded_value)));
    }
}

void quantize_q16a_tr_reference(const float *x, uint8_t *input_quantized, float input_scale, int32_t input_zp, int64_t k) {
    (void)input_zp;
    
    // Calculate inverse scale to avoid division in the loop
    const float inverse_scale = (input_scale != 0.0f) ? (1.0f / input_scale) : 0.0f;
    int16_t *input_quantized_q16 = (int16_t*)input_quantized;
     // Quantize each element: quantized_value = round(float_value / scale) 
    for (int64_t i = 0; i < k; i++) {
        const float scaled_value = x[i] * inverse_scale;
        const int rounded_value = static_cast<int>(std::round(scaled_value));
        
        // Clamp to int16_t range [int16 min, int16 max]
        input_quantized_q16[i] = static_cast<int16_t>(std::max(INT16_MIN, std::min(INT16_MAX, rounded_value)));
    }
}

#if defined(__ARM_NEON)
// this intrisic is provided on ARMv8-A(aarch64)
inline static int32_t vaddvq_s32(int32x4_t v) {
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}
#endif

/**
 * @brief Vector dot product for TRIX quantized weights (4-bit) and activations (8-bit)
 * 
 * This function computes the dot product between 4-bit quantized weights and 8-bit quantized inputs
 * for 32 output channels. It uses NEON intrinsics for ARM optimization when available.
 * 
 * @param n Number of input elements (must be multiple of 32)
 * @param s Output array for 32 dot product results
 * @param w_ptr Pointer to 4-bit quantized weight data
 * @param w_scales Per-channel scale factors for weights
 * @param w_zerops Per-channel zero points for weights
 * @param i_ptr Pointer to 8-bit quantized input data
 * @param i_scale Scale factor for input quantization
 * @param i_zerop Zero point for input quantization
 */
void vec_dot_q4w_tr_q8a_tr(int n, float *s /*output*/, const uint8_t *w_ptr, const float *w_scales, const uint8_t *w_zerops, const uint8_t *i_ptr, float i_scale, uint32_t i_zerop)
{
    // Constants
    constexpr int NUM_CHANNELS = 32;
    constexpr int ELEMENTS_PER_CHANNEL = 16;  // 32 elements per channel, 2 elements per byte (4-bit)
    
    
    assert(n % NUM_CHANNELS == 0);

#if defined(__ARM_NEON)
    constexpr int ACCUMULATION_INTERVAL = 4;  // Accumulate every 4 iterations to prevent overflow
    // NEON-optimized implementation for ARM processors
    
    // Extract quantization parameters
    const uint8_t input_zp = static_cast<uint8_t>(i_zerop);
    const float input_scale = i_scale;
    const uint8_t *input_data = i_ptr;
    const uint8_t *weight_data = w_ptr;  // 4-bit quantized values (n/2 bytes)
    
    const int num_blocks = n / NUM_CHANNELS;

    // Accumulation registers for each channel
    int32x4_t sum32_low[NUM_CHANNELS];   // Lower 32-bit accumulators
    int32x4_t sum32_high[NUM_CHANNELS];  // Higher 32-bit accumulators
    int16x8_t sum16_low[NUM_CHANNELS];   // Lower 16-bit accumulators
    int16x8_t sum16_high[NUM_CHANNELS];  // Higher 16-bit accumulators
    
    // Initialize accumulators to zero
    for (int channel = 0; channel < NUM_CHANNELS; ++channel) {
        sum32_low[channel] = vdupq_n_s32(0);
        sum32_high[channel] = vdupq_n_s32(0);
        sum16_low[channel] = vdupq_n_s16(0);
        sum16_high[channel] = vdupq_n_s16(0);
    }

    // NEON constants
    const uint8x16_t mask_4bit = vdupq_n_u8(0x0F);  // Mask to extract lower 4 bits
    const int16x8_t input_zp_vec = vdupq_n_s16(input_zp);

    // Process blocks of 32 elements
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint8_t *current_input = input_data + (NUM_CHANNELS * block_idx);

        // Load 32 input elements: 16 in lower half, 16 in upper half
        const uint8x16_t input_low = vld1q_u8(current_input);
        const uint8x16_t input_high = vld1q_u8(current_input + 16);

        // Convert to int16 and subtract zero point
        const int16x8_t input_ll = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_low))), input_zp_vec);
        const int16x8_t input_lh = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_low))), input_zp_vec);
        const int16x8_t input_hl = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_high))), input_zp_vec);
        const int16x8_t input_hh = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_high))), input_zp_vec);

        // Process each channel
        for (int channel = 0; channel < NUM_CHANNELS; ++channel) {
            // Load 16 bytes of 4-bit weights for this channel
            const uint8x16_t packed_weights = vld1q_u8(weight_data + block_idx * (NUM_CHANNELS * ELEMENTS_PER_CHANNEL) + channel * ELEMENTS_PER_CHANNEL);

            // Unpack 4-bit weights to 8-bit: separate even and odd elements
            const int8x16_t weight_even = vreinterpretq_s8_u8(vandq_u8(packed_weights, mask_4bit));
            const int8x16_t weight_odd = vreinterpretq_s8_u8(vshrq_n_u8(packed_weights, 4));

            // Subtract weight zero point
            const int8x16_t weight_zp_vec = vdupq_n_s8(w_zerops[channel]);
            const int8x16_t weight_even_sub = vsubq_s8(weight_even, weight_zp_vec);
            const int8x16_t weight_odd_sub = vsubq_s8(weight_odd, weight_zp_vec);

            // Interleave even and odd weights to restore original order
            const int8x16x2_t weight_interleaved = vzipq_s8(weight_even_sub, weight_odd_sub);
            const int8x16_t weight_low = weight_interleaved.val[0];
            const int8x16_t weight_high = weight_interleaved.val[1];

            // Convert to int16 for multiplication
            const int16x8_t weight_ll = vmovl_s8(vget_low_s8(weight_low));
            const int16x8_t weight_lh = vmovl_s8(vget_high_s8(weight_low));
            const int16x8_t weight_hl = vmovl_s8(vget_low_s8(weight_high));
            const int16x8_t weight_hh = vmovl_s8(vget_high_s8(weight_high));

            // Multiply-accumulate: sum += weight * input
            sum16_low[channel] = vmlaq_s16(vmlaq_s16(sum16_low[channel], weight_ll, input_ll), weight_lh, input_lh);
            sum16_high[channel] = vmlaq_s16(vmlaq_s16(sum16_high[channel], weight_hl, input_hl), weight_hh, input_hh);
        }

        // Periodically accumulate 16-bit results into 32-bit to prevent overflow
        if (block_idx % ACCUMULATION_INTERVAL == ACCUMULATION_INTERVAL - 1) {
            for (int channel = 0; channel < NUM_CHANNELS; ++channel) {
                sum32_low[channel] = vaddq_s32(sum32_low[channel], vpaddlq_s16(sum16_low[channel]));
                sum32_high[channel] = vaddq_s32(sum32_high[channel], vpaddlq_s16(sum16_high[channel]));
                sum16_low[channel] = vdupq_n_s16(0);
                sum16_high[channel] = vdupq_n_s16(0);
            }
        }
    }

    // Final accumulation for any remaining 16-bit results
    if (num_blocks % ACCUMULATION_INTERVAL != 0) {
        for (int channel = 0; channel < NUM_CHANNELS; ++channel) {
            sum32_low[channel] = vaddq_s32(sum32_low[channel], vpaddlq_s16(sum16_low[channel]));
            sum32_high[channel] = vaddq_s32(sum32_high[channel], vpaddlq_s16(sum16_high[channel]));
        }
    }

    // Combine high and low accumulators, apply scaling, and store results
    for (int channel = 0; channel < NUM_CHANNELS; ++channel) {
        const int32x4_t combined_sum = vaddq_s32(sum32_low[channel], sum32_high[channel]);
        const int32_t final_sum = vaddvq_s32(combined_sum);
        s[channel] = static_cast<float>(final_sum) * w_scales[channel] * input_scale;
    }

#else
    // Reference implementation for non-ARM platforms
    
    // Extract quantization parameters
    const uint8_t input_zp = static_cast<uint8_t>(i_zerop);
    const float input_scale = i_scale;
    const uint8_t *input_data = i_ptr;
    const uint8_t *weight_data = w_ptr;  // 4-bit quantized values (n/2 bytes)

    const int num_blocks = n / NUM_CHANNELS;
    int32_t channel_accumulators[NUM_CHANNELS] = {0};

    const uint8_t *current_input;
    const uint8_t *current_weight;

    // Process each block
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        current_input = input_data + (NUM_CHANNELS * block_idx);

        // Process each channel
        for (int channel = 0; channel < NUM_CHANNELS; ++channel) {
            current_weight = weight_data + block_idx * (NUM_CHANNELS * ELEMENTS_PER_CHANNEL) + channel * ELEMENTS_PER_CHANNEL;
            
            // Compute dot product for 32 inputs and 32 weights (4-bit each)
            for (int elem_idx = 0; elem_idx < ELEMENTS_PER_CHANNEL; ++elem_idx) {
                // Unpack 4-bit weights: extract lower and upper 4 bits
                const int32_t weight_low = (current_weight[elem_idx] & 0x0F) - w_zerops[channel];
                const int32_t weight_high = (current_weight[elem_idx] >> 4) - w_zerops[channel];

                // Extract corresponding input values
                const int32_t input_low = current_input[2 * elem_idx] - input_zp;
                const int32_t input_high = current_input[2 * elem_idx + 1] - input_zp;

                // Accumulate products
                channel_accumulators[channel] += (weight_low * input_low) + (weight_high * input_high);
            }
        }
    }

    // Apply scaling and store results
    for (int channel = 0; channel < NUM_CHANNELS; ++channel) {
        s[channel] = static_cast<float>(channel_accumulators[channel]) * w_scales[channel] * input_scale;
    }

#endif
}


/**
 * @brief Alternative vector dot product for 8-bit weights and 8-bit activations
 * 
 * This is an example alternative implementation that could use different algorithms
 * or optimizations compared to vec_dot_q4w_tr_q8a_tr. Currently implemented as a
 * placeholder that delegates to the original function for compatibility.
 * 
 * @param n Number of input elements (must be multiple of 32)
 * @param s Output array for 32 dot product results
 * @param w_ptr Pointer to 8-bit quantized weight data
 * @param w_scales Per-channel scale factors for weights
 * @param w_zerops Per-channel zero points for weights
 * @param i_ptr Pointer to 8-bit quantized input data
 * @param i_scale Scale factor for input quantization
 * @param i_zerop Zero point for input quantization
 */
void vec_dot_q8w_tr_q8a_tr(int n, float *s /*output*/, const uint8_t *w_ptr, const float *w_scales, const uint8_t *w_zerops, const uint8_t *i_ptr, float i_scale, uint32_t i_zerop)
{
    assert(n % 32 == 0);

    // number of channels
    const int nc = 32;

#if defined(__ARM_NEON)
    uint8_t input_zp = i_zerop;
    float input_scale = iqp->scale;
    const uint8_t *y = i_ptr;

    // weight setting
    const uint8_t *x = w_ptr; // n 8bit quantize values
    

    int nb = n / 32;

    int32x4_t vsum[nc][4];
    for (int j = 0; j < nc; ++ j) {
        for (int k = 0; k < 4; ++ k) {
            vsum[j][k] = vdupq_n_s32(0);
        }
    }

    const int16x8_t vinput_zp = vdupq_n_s16(input_zp);

    for (int i = 0; i < nb; ++ i) {
        int16x4_t input[8];

        for (int k = 0; k < 2; ++ k) {
            const uint8_t *input_ptr = y + 32 * i + 16 * k;

            // load inputs
            const uint8x16_t cur_input = vld1q_u8(input_ptr);

            // 8bit -> 16bit
            const int16x8_t input_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cur_input)));
            const int16x8_t input_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(cur_input)));

            // subtract zp
            const int16x8_t input_ls = vsubq_s16(input_l, vinput_zp);
            const int16x8_t input_hs = vsubq_s16(input_h, vinput_zp);

            // split
            input[4 * k + 0] = vget_low_s16(input_ls);
            input[4 * k + 1] = vget_high_s16(input_ls);
            input[4 * k + 2] = vget_low_s16(input_hs);
            input[4 * k + 3] = vget_high_s16(input_hs);
        }

        for (int j = 0; j < nc; ++ j){  // j is channel index
            for (int k = 0; k < 2; ++ k) {
                // load weights
                const uint8x16_t weight = vld1q_u8(x + (32 * nc) * i + 32 * j + 16 * k);

                // 8bit -> 16bit
                const int16x8_t weight_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(weight)));
                const int16x8_t weight_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(weight)));

                // subtract zp
                const int16x8_t weight_zp = vdupq_n_s16(w_zerops[j]);
                const int16x8_t weight_ls = vsubq_s16(weight_l, weight_zp);
                const int16x8_t weight_hs = vsubq_s16(weight_h, weight_zp);

                // split
                const int16x4_t weight_ll = vget_low_s16(weight_ls);
                const int16x4_t weight_lh = vget_high_s16(weight_ls);
                const int16x4_t weight_hl = vget_low_s16(weight_hs);
                const int16x4_t weight_hh = vget_high_s16(weight_hs);

                // sum += weight * input
                vsum[j][0] = vmlal_s16(vsum[j][0], vget_low_s16(weight_ls), input[4 * k + 0]);
                vsum[j][1] = vmlal_s16(vsum[j][1], vget_high_s16(weight_ls), input[4 * k + 1]);
                vsum[j][2] = vmlal_s16(vsum[j][2], vget_low_s16(weight_hs), input[4 * k + 2]);
                vsum[j][3] = vmlal_s16(vsum[j][3], vget_high_s16(weight_hs), input[4 * k + 3]);
            }
        }
    }

    for (int j = 0; j < nc; ++ j) {
        s[j] = vaddvq_s32(vaddq_s32(vaddq_s32(vsum[j][0], vsum[j][1]), vaddq_s32(vsum[j][2], vsum[j][3]))) * w_scales[j] * input_scale;
    }
#else
    // Reference implementation

    // input(vy) setting
    uint8_t input_zp = i_zerop;
    float input_scale = i_scale;
    const uint8_t *y = i_ptr;

    // weight(vx) setting
    const uint8_t *x = w_ptr; // n 8bit quantize values


    int nb = n / 32;
    int64_t sum[32] = {0,};

    uint8_t *cur_input;
    uint8_t *cur_weight;

    for (int i = 0; i < nb; ++ i) {
        cur_input = (uint8_t *)(y + 32 * i); // input update for next block

        for (int j = 0; j < nc; ++ j){  // j is channel index
            cur_weight = (uint8_t *)(x + i * (nc * 32) + j * 32);

            for (int k = 0; k < 32; ++ k) {  // dot product for 32 inputs and 32 weights
                sum[j] += (cur_weight[k] - w_zerops[j]) * (cur_input[k] - input_zp);
            }
        }
    }

    for (int j = 0; j < nc; ++ j){
        s[j] = sum[j] * w_scales[j] * input_scale;
    }
#endif


}

void vec_dot_q8w_tr_q16a_tr(int n, float *s /*output*/, const uint8_t *w_ptr, const float *w_scales, const uint8_t *w_zerops, const uint8_t *i_ptr, float i_scale, uint32_t i_zerop)
{
    (void)i_zerop;
    assert(n % 16 == 0);

    // number of channels
    const int nc = 16;

#if defined(__ARM_NEON)
    // input(vy) setting
    float input_scale = i_scale;
    const int16_t *y = i_ptr;
    // weight(vx) setting
    const uint8_t *x = w_ptr; // n 8bit quantize values
    int nb = n / 16;

    int16_t * cur_input;
    uint8_t * cur_weight;

    int32x4_t sum32[nc];
    for (int j = 0; j < nc; ++ j) sum32[j] = vdupq_n_s32(0);

    for (int i = 0; i < nb; ++ i) {
        cur_input = (int16_t *)(y + 16 * i); // input update for next block

        // load inputs
        const int16x8_t input_l = vld1q_s16(cur_input);
        const int16x8_t input_h = vld1q_s16(cur_input + 8);

        const int16x4_t input_ll = vget_low_s16(input_l);
        const int16x4_t input_lh = vget_high_s16(input_l);
        const int16x4_t input_hl = vget_low_s16(input_h);
        const int16x4_t input_hh = vget_high_s16(input_h);

        for (int j = 0; j < nc; ++ j){  // j is channel index
            // load weights
            const uint8x16_t weight = vld1q_u8(x + i * (nc * 16) + j * 16);
            const int16x8_t weight_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(weight)));
            const int16x8_t weight_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(weight)));

            // subtract zp
            const int16x8_t weight_zp = vdupq_n_s16(w_zerops[j]);
            const int16x8_t weight_ls = vsubq_s16(weight_l, weight_zp);
            const int16x8_t weight_hs = vsubq_s16(weight_h, weight_zp);

            // suml[j] += weight*input
            const int16x4_t weight_ll = vget_low_s16(weight_ls);
            const int16x4_t weight_lh = vget_high_s16(weight_ls);
            const int16x4_t weight_hl = vget_low_s16(weight_hs);
            const int16x4_t weight_hh = vget_high_s16(weight_hs);

            sum32[j] = vmlal_s16(vmlal_s16(sum32[j], weight_ll, input_ll), weight_lh, input_lh);
            sum32[j] = vmlal_s16(vmlal_s16(sum32[j], weight_hl, input_hl), weight_hh, input_hh);
        }
    }

    for (int j = 0; j < nc; ++ j){
        // s[j] = (vgetq_lane_s32(sum32[j], 0) + vgetq_lane_s32(sum32[j], 1) + vgetq_lane_s32(sum32[j], 2) + vgetq_lane_s32(sum32[j], 3)) * qp[j].scale * input_scale;
        s[j] = vaddvq_s32(sum32[j]) * w_scales[j] * input_scale;
    }

#else
    // Reference implementation

    // input(vy) setting
    float input_scale = i_scale;
    const int16_t *y = (const int16_t *)i_ptr;
    // weight(vx) setting
    const uint8_t *x = w_ptr; // n 8bit quantize values
  
    int nb = n / 16;
    int64_t sum[16] = {0,};

    int16_t * cur_input;
    uint8_t * cur_weight;

    for (int i = 0; i < nb; ++ i) {
        cur_input = (int16_t *)(y + 16 * i); // input update for next block

        for (int j = 0; j < nc; ++ j){  // j is channel index
            cur_weight = (uint8_t *)(x + i * (nc * 16) + j * 16);

            for (int k = 0; k < 16; ++ k) {  // dot product for 16 inputs and 16 weights
                sum[j] += (cur_weight[k] - w_zerops[j]) * cur_input[k]; 
             }
        }
    }

    for (int j = 0; j < nc; ++ j){
        s[j] = sum[j] * w_scales[j] * input_scale;
    }
#endif

}

/**
 * @brief Function pointer type for vector dot product operations
 * 
 * This defines the signature for vector dot product functions that can be used
 * with the templated FullyConnectedTRIXW4A8 implementation.
 * 
 * @param n Number of elements to process
 * @param s Output array for dot product results
 * @param w_ptr Pointer to weight data
 * @param w_scales Per-channel scale factors for weights
 * @param w_zerops Per-channel zero points for weights
 * @param i_ptr Pointer to input data
 * @param i_scale Scale factor for input quantization
 * @param i_zerop Zero point for input quantization
 */
using VecDotFunction = void(*)(int n, float *s, const uint8_t *w_ptr, const float *w_scales, 
                              const uint8_t *w_zerops, const uint8_t *i_ptr, float i_scale, uint32_t i_zerop);

using QuantizeFunction = void(*)(const float *x, uint8_t *input_quantized, float input_scale, int32_t input_zp, int64_t k); 

/**
 * @brief Template implementation for FullyConnected layer with TRIX quantization
 * 
 * This is a templated implementation that allows different vector dot product functions
 * to be injected, avoiding code duplication while maintaining the same functionality.
 * 
 * @tparam VecDotFunc The vector dot product function to use
 * @param params FullyConnected parameters (currently unused)
 * @param input_shape Shape of input tensor
 * @param input_data Input tensor data (float32, will be quantized to uint8)
 * @param filter_shape Shape of filter/weights tensor
 * @param filter_data Filter data (4-bit quantized)
 * @param bias_shape Shape of bias tensor
 * @param bias_data Bias data (int32)
 * @param output_shape Shape of output tensor
 * @param output_data Output tensor data (currently unused - output written to float buffer)
 * @param in_ch_stride Input channel stride for processing
 * @param input_scale Scale factor for input quantization
 * @param input_zp Zero point for input quantization
 * @param offset Offset array for weight data access
 * @param filter_per_channel_scales Per-channel scale factors for weights
 * @param filter_per_channel_zp Per-channel zero points for weights
 */
template<VecDotFunction VecDotFunc, QuantizeFunction QuantFunc,int MICRO_KERNEL_SIZE, typename InputQType>
inline void FullyConnectedTRIXImpl(FullyConnectedParams &params,
                                       const Shape &input_shape, const uint8_t *input_data,
                                       const Shape &filter_shape, const uint8_t *filter_data,
                                       const Shape &bias_shape, const int32_t *bias_data,
                                       const Shape &output_shape, uint8_t *output_data, 
                                       int32_t in_ch_stride,
                                       float input_scale,
                                       int32_t input_zp, const std::vector<int32_t> &offset,
                                       const float *filter_per_channel_scales,
                                       const int32_t *filter_per_channel_zp)
{
  // Extract dimensions
  const int total_input_size = input_shape.FlatSize();
  const int input_size = filter_shape.Dims(1);
  const int input_row_size = total_input_size / input_size;  // Handle 3D input e.g [1x128x512]
  const int output_channels = filter_shape.Dims(0);
  const int weight_cols = filter_shape.Dims(1);

  // Step 1: Quantize F32 input to Q8/Q16 using input scale and zero point
  std::vector<InputQType> input_quantized(total_input_size);
  QuantFunc(reinterpret_cast<const float*>(input_data), 
            reinterpret_cast<uint8_t*>(input_quantized.data()), 
                           input_scale, 
                           input_zp, 
                           total_input_size);

  // Step 2: Prepare uint8_t type per-channel zero points for micro kernel
  std::vector<uint8_t> filter_zerop(output_channels);
  for (int i = 0; i < output_channels; i++) {
    filter_zerop[i] = static_cast<uint8_t>(filter_per_channel_zp[i]);
  }
  
  // Step 3: Main computation loop
  // Micro kernel processes W[32, emb_size] x I[1, emb_size] -> O[32]
  // Each iteration produces 32 outputs along the output channel axis
  std::vector<float> output_buffer(MICRO_KERNEL_SIZE);
  
  // Process each input row
  for (int32_t input_row_idx = 0; input_row_idx < input_row_size; input_row_idx++) {
    
    // Process output channels in blocks of MICRO_KERNEL_SIZE
    for (int32_t output_channel_start = 0; output_channel_start < output_channels; output_channel_start += MICRO_KERNEL_SIZE) {
      
      // Get destination pointer for this block
      float *destination_ptr = reinterpret_cast<float*>(output_data) + 
                              (input_row_idx * output_channels) + 
                              output_channel_start;
      
      // Process input channels in strides
      for (int32_t input_channel_start = 0; input_channel_start < weight_cols; input_channel_start += in_ch_stride) {
        
        // Calculate offset index for weight data access
        const size_t offset_index = (output_channel_start / MICRO_KERNEL_SIZE) * 
                                   ((weight_cols + in_ch_stride - 1) / in_ch_stride) + 
                                   ((input_channel_start + in_ch_stride - 1) / in_ch_stride);
        
        // Get pointers to current weight and input data
        const uint8_t *weight_ptr = filter_data + offset[offset_index];
        const uint8_t *current_filter_zp = filter_zerop.data() + output_channel_start;
        const float *current_filter_scale = filter_per_channel_scales + output_channel_start;
        const uint8_t *input_ptr = reinterpret_cast<const uint8_t*>(input_quantized.data()) + 
                                  (weight_cols * input_row_idx) + 
                                  input_channel_start;
        
        // Calculate actual stride (handle edge case)
        const size_t actual_stride = std::min(static_cast<size_t>(in_ch_stride), 
                                             static_cast<size_t>(weight_cols - input_channel_start));
        
        // Execute micro kernel: compute dot product for this block using the templated function
        VecDotFunc(actual_stride, 
                  output_buffer.data(), 
                  weight_ptr, 
                  current_filter_scale, 
                  current_filter_zp, 
                  input_ptr, 
                  input_scale, 
                  input_zp);
        
        // Initialize destination buffer on first stride
        if (input_channel_start == 0) {
          std::memset(destination_ptr, 0, sizeof(float) * MICRO_KERNEL_SIZE);
        }
        
        // Accumulate results
        for (int i = 0; i < MICRO_KERNEL_SIZE; i++) {
          destination_ptr[i] += output_buffer[i];
        }
      }
    }
  }
  
  // Suppress unused parameter warnings
  (void)params;
  (void)bias_shape;
  (void)bias_data;
  (void)output_shape;
  (void)output_data;
  (void)offset;
  (void)filter_per_channel_scales;
  (void)filter_per_channel_zp;
}

/**
 * @brief FullyConnected layer with TRIX W4A8 quantization (4-bit weights, 8-bit activations)
 * 
 * This function implements a fully connected layer with TRIX quantization where:
 * - Weights are quantized to 4-bit per channel
 * - Activations are quantized to 8-bit
 * - Uses micro-kernel approach processing 32 output channels at a time
 * - Uses the original vec_dot_q4w_tr_q8a_tr function
 * 
 * @param params FullyConnected parameters (currently unused)
 * @param input_shape Shape of input tensor
 * @param input_data Input tensor data (float32, will be quantized to uint8)
 * @param filter_shape Shape of filter/weights tensor
 * @param filter_data Filter data (4-bit quantized)
 * @param bias_shape Shape of bias tensor
 * @param bias_data Bias data (int32)
 * @param output_shape Shape of output tensor
 * @param output_data Output tensor data (currently unused - output written to float buffer)
 * @param in_ch_stride Input channel stride for processing
 * @param input_scale Scale factor for input quantization
 * @param input_zp Zero point for input quantization
 * @param offset Offset array for weight data access
 * @param filter_per_channel_scales Per-channel scale factors for weights
 * @param filter_per_channel_zp Per-channel zero points for weights
 */
inline void FullyConnectedTRIXW4A8(FullyConnectedParams &params,
                                   const Shape &input_shape, const uint8_t *input_data,
                                   const Shape &filter_shape, const uint8_t *filter_data,
                                   const Shape &bias_shape, const int32_t *bias_data,
                                   const Shape &output_shape, uint8_t *output_data, 
                                   int32_t in_ch_stride,
                                   float input_scale,
                                   int32_t input_zp, const std::vector<int32_t> &offset,
                                   const float *filter_per_channel_scales,
                                   const int32_t *filter_per_channel_zp)
{
  // Use the template implementation with the original vec_dot_q4w_tr_q8a_tr function
  FullyConnectedTRIXImpl<vec_dot_q4w_tr_q8a_tr, quantize_q8a_tr_reference, 32, uint8_t>(params, input_shape, input_data, filter_shape, 
                                                   filter_data, bias_shape, bias_data, output_shape, 
                                                   output_data, in_ch_stride, input_scale, input_zp, 
                                                   offset, filter_per_channel_scales, filter_per_channel_zp);
}

inline void FullyConnectedTRIXW8A8(FullyConnectedParams &params,
                                   const Shape &input_shape, const uint8_t *input_data,
                                   const Shape &filter_shape, const uint8_t *filter_data,
                                   const Shape &bias_shape, const int32_t *bias_data,
                                   const Shape &output_shape, uint8_t *output_data, 
                                   int32_t in_ch_stride,
                                   float input_scale,
                                   int32_t input_zp, const std::vector<int32_t> &offset,
                                   const float *filter_per_channel_scales,
                                   const int32_t *filter_per_channel_zp)
{
  // Use the template implementation with the original vec_dot_q4w_tr_q8a_tr function
  FullyConnectedTRIXImpl<vec_dot_q8w_tr_q8a_tr, quantize_q8a_tr_reference, 32, uint8_t>(params, input_shape, input_data, filter_shape, 
                                                   filter_data, bias_shape, bias_data, output_shape, 
                                                   output_data, in_ch_stride, input_scale, input_zp, 
                                                   offset, filter_per_channel_scales, filter_per_channel_zp);
}

inline void FullyConnectedTRIXW8A16(FullyConnectedParams &params,
                                   const Shape &input_shape, const uint8_t *input_data,
                                   const Shape &filter_shape, const uint8_t *filter_data,
                                   const Shape &bias_shape, const int32_t *bias_data,
                                   const Shape &output_shape, uint8_t *output_data, 
                                   int32_t in_ch_stride,
                                   float input_scale,
                                   int32_t input_zp, const std::vector<int32_t> &offset,
                                   const float *filter_per_channel_scales,
                                   const int32_t *filter_per_channel_zp)
{
  // Use the template implementation with the original vec_dot_q4w_tr_q8a_tr function
  FullyConnectedTRIXImpl<vec_dot_q8w_tr_q16a_tr, quantize_q16a_tr_reference, 16, int16_t>(params, input_shape, input_data, filter_shape, 
                                                   filter_data, bias_shape, bias_data, output_shape, 
                                                   output_data, in_ch_stride, input_scale, input_zp, 
                                                   offset, filter_per_channel_scales, filter_per_channel_zp);
}

inline void FullyConnectedSparseWeightRandom(
  const FullyConnectedParams &params, [[maybe_unused]] const Shape &input_shape,
  const float *input_data, const Shape &weights_shape, const float *weights_data,
  [[maybe_unused]] const Shape &bias_shape, const float *bias_data, const Shape &output_shape,
  float *output_data, const uint16_t *w1_segments, const uint16_t *w1_indices)
{

  assert(weights_shape.DimensionsCount() == 2);
  assert(output_shape.DimensionsCount() == 2);

  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1);
  const int output_depth =
    MatchingDim(weights_shape, weights_dims_count - 2, output_shape, output_dims_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dims_count - 1);

  if (bias_data)
  {
    VectorBatchVectorAssign(bias_data, output_depth, batches, output_data);
  }
  else
  {
    ZeroVector(output_data, batches * output_depth);
  }
  for (int b = 0; b < batches; ++b)
  {
    for (int idx_0 = 0; idx_0 < output_depth; ++idx_0)
    {
      for (int pw1 = w1_segments[idx_0]; pw1 < w1_segments[idx_0 + 1]; ++pw1)
      {
        int idx_1 = w1_indices[pw1];
        output_data[b * output_depth + idx_0] +=
          weights_data[pw1] * input_data[b * accum_depth + idx_1];
      }
    }
  }
  if (params.activation != FusedActivationFunctionType::kNone)
  {
    // Apply activation function
    ApplyActivationToVector(output_data, batches * output_depth, params.activation, output_data);
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_FULLY_CONNECTED_H__
