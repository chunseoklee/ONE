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

void quantize_q8a_tr_reference(const float *x, uint8_t *input_quantized, float input_scale, int32_t input_zp, int64_t k) {
    const float id = input_scale ? 1.0f/input_scale : 0.0f;
    for (int64_t i = 0; i < k; i++) {
        input_quantized[i] = std::min(UINT8_MAX, std::max(0, static_cast<int>(std::round(x[i] * id + input_zp))));
    }
}

#if defined(__ARM_NEON)
// this intrisic is provided on ARMv8-A(aarch64)
inline static int32_t vaddvq_s32(int32x4_t v) {
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}
#endif

void vec_dot_q4w_tr_q8a_tr(int n, float *s /*output*/, const uint8_t *w_ptr, const float *w_scales, const uint8_t *w_zerops, const uint8_t *i_ptr, float i_scale, uint32_t i_zerop)
{
    assert(n % 32 == 0);

#if defined(__ARM_NEON)
    // number of channels
    const int nc = 32;
    // input setting
    // struct ggml_trix_qparams * iqp = vy->d;
    uint8_t input_zp = i_zerop;
    float input_scale = i_scale;
    const uint8_t *y = i_ptr;
    // weight setting
    const uint8_t *x = w_ptr; // n 4bit quantize values ( n/2 byte)
  
    int nb = n / 32;

    uint8_t * cur_input;
    //uint8_t * cur_weight;

    int32x4_t sum32l[nc];
    int32x4_t sum32h[nc];
    int16x8_t sum16l[nc];
    int16x8_t sum16h[nc];
    for (int j = 0; j < nc; ++ j) {
        sum32l[j] = vdupq_n_s32(0);
        sum32h[j] = vdupq_n_s32(0);
        sum16l[j] = vdupq_n_s16(0);
        sum16h[j] = vdupq_n_s16(0);
    }

    const uint8x16_t mask_lower_4 = vdupq_n_u8(0x0F);
    const int16x8_t vinput_zp = vdupq_n_s16(input_zp);

    for (int i = 0; i < nb; ++ i) {
        cur_input = (uint8_t *)(y + 32*i); // input update for next block

        // load inputs, input_l contains first 16 elements and input_h contains next 16 elements
        const uint8x16_t input_l = vld1q_u8(cur_input);
        const uint8x16_t input_h = vld1q_u8(cur_input + 16);

        const int16x8_t input_ll = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_l))), vinput_zp);
        const int16x8_t input_lh = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_l))), vinput_zp);
        const int16x8_t input_hl = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_h))), vinput_zp);
        const int16x8_t input_hh = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_h))), vinput_zp);

        for (int j = 0; j < nc; ++ j){  // j is channel index
            // load weights
            const uint8x16_t weight = vld1q_u8(x + i*(nc*16) + j*16);

            // 4bit -> 8bit, weight0 contains even elements and weight 1 contains odd elements
            const int8x16_t weight0 = vreinterpretq_s8_u8(vandq_u8(weight, mask_lower_4));
            const int8x16_t weight1 = vreinterpretq_s8_u8(vshrq_n_u8(weight, 4));

            // subtract zp
            const int8x16_t weight_zp = vdupq_n_s8(w_zerops[j]);
            const int8x16_t weight0s = vsubq_s8(weight0, weight_zp);
            const int8x16_t weight1s = vsubq_s8(weight1, weight_zp);

            // zip weights, weight_l contains first 16 elements and weight_h contains next 16 elements
            const int8x16x2_t weight_zipped = vzipq_s8(weight0s, weight1s);
            const int8x16_t weight_l = weight_zipped.val[0];
            const int8x16_t weight_h = weight_zipped.val[1];

            const int16x8_t weight_ll = vmovl_s8(vget_low_s8(weight_l));
            const int16x8_t weight_lh = vmovl_s8(vget_high_s8(weight_l));
            const int16x8_t weight_hl = vmovl_s8(vget_low_s8(weight_h));
            const int16x8_t weight_hh = vmovl_s8(vget_high_s8(weight_h));

            // sum[j] += weight*input
            sum16l[j] = vmlaq_s16(vmlaq_s16(sum16l[j], weight_ll, input_ll), weight_lh, input_lh);
            sum16h[j] = vmlaq_s16(vmlaq_s16(sum16h[j], weight_hl, input_hl), weight_hh, input_hh);
        }

        if(i % 4 == 3){
            for (int j = 0; j < nc; ++ j){
                // sum32 += sum16 and set sum16 to 0
                sum32l[j] = vaddq_s32(sum32l[j], vpaddlq_s16(sum16l[j]));
                sum32h[j] = vaddq_s32(sum32h[j], vpaddlq_s16(sum16h[j]));
                sum16l[j] = vdupq_n_s16(0);
                sum16h[j] = vdupq_n_s16(0);
            }
        }
    }

    if(nb % 4 != 0){
        for (int j = 0; j < nc; ++ j){
            // sum32 += sum16
            sum32l[j] = vaddq_s32(sum32l[j], vpaddlq_s16(sum16l[j]));
            sum32h[j] = vaddq_s32(sum32h[j], vpaddlq_s16(sum16h[j]));
        }
    }

    for (int j = 0; j < nc; ++j){
        s[j] = vaddvq_s32(vaddq_s32(sum32l[j], sum32h[j]))* w_scales[j] * input_scale;
    } 

#else
    // Reference implementation

    uint8_t input_zp = i_zerop;
    float input_scale = i_scale;
    const uint8_t *y = i_ptr; // In this function, y is input ptr

    // weight setting
    const uint8_t *x = w_ptr; // n 4bit quantize values ( n/2 byte)
    //const struct ggml_trix_qparams * qp = vx->d; // n qparam(zp, scale)

    int nb = n / 32;
    int32_t sum[32] = {0,};

    const uint8_t * cur_input;
    const uint8_t * cur_weight;

    for (int i = 0; i < nb; i++) {
        cur_input = y + 32*i; // input update for next block

        for (int j = 0; j < 32; ++j){  // j is channel index
            cur_weight = x + i*(32*16) + j*16;
            for (int k = 0; k < 16; ++k) {  // dot product for 32 inputs and 32 weights
                // 4bit -> 8bit
                const int32_t v0 = (cur_weight[k] & 0x0F) - w_zerops[j];
                const int32_t v1 = (cur_weight[k] >>   4) - w_zerops[j];

                const int32_t i0 = cur_input[2*k] - input_zp;
                const int32_t i1 = cur_input[2*k + 1] - input_zp;

                sum[j] += (v0 * i0) + (v1 * i1);
            }
        }
    }

    for (int j = 0; j < 32; ++j){
        s[j] = sum[j] * w_scales[j] * input_scale;
    }

#endif

}


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
  // 1. Quantize F32 input to Q8 using input scale and zero point
  std::vector<uint8_t> input_quantized(input_shape.FlatSize());
  quantize_q8a_tr_reference(reinterpret_cast<const float*>(input_data), input_quantized.data(), input_scale, input_zp,input_shape.FlatSize());

  // 2. Main loop
  
  // 2-1. convert weight zerop to uint8_t* for micro kernel
  std::vector<uint8_t> filter_zerop(filter_shape.Dims(0)); // per channel zerop as uint8_t for micro kernel
  for (int i = 0; i < filter_shape.Dims(0); i++) {
    filter_zerop[i] = filter_per_channel_zp[i];
  }
  
  // 2-2. micro kernel
  // Our compute unit for micro kernel is W[32, emb_size] x I[1, emb_size] -> O[32]
  // Thus, each loop produces 32 outputs(along output channel axis)

  std::vector<float> out32(32);
  const uint8_t *w_ptr = nullptr;  

  auto begin_row0 = 0;
  auto end_row0 = filter_shape.Dims(0);
  auto blck_0 = 32;
  int32_t weight_col_size = filter_shape.Dims(1);
  const uint8_t *i_ptr = nullptr;
  
  for (int32_t iir0 = begin_row0; iir0 < end_row0; iir0 += blck_0) {
    float * dst_col = (float *) ((float*)output_data + iir0);
    for (int32_t in_ch = 0; in_ch < weight_col_size; in_ch += in_ch_stride) {
      size_t off_index = (iir0/blck_0)*((weight_col_size+in_ch_stride-1)/in_ch_stride) + (in_ch+in_ch_stride-1)/in_ch_stride;
      w_ptr = filter_data + offset[off_index];
      const uint8_t *cur_filter_zp = filter_zerop.data() + iir0;
      const float *cur_filter_scale = filter_per_channel_scales + iir0;
      i_ptr = input_quantized.data() + in_ch;
      
      size_t real_stride= std::min(in_ch_stride, weight_col_size-in_ch);
      vec_dot_q4w_tr_q8a_tr(real_stride, out32.data(), w_ptr, cur_filter_scale, cur_filter_zp, i_ptr, input_scale, input_zp);
      // 32 or 16 float result acumulated to dst_col
      if (in_ch == 0) {
        memset(dst_col, 0, sizeof(float)*blck_0);
      }
      for(int i = 0; i < blck_0; i++) {
        *(dst_col+i) += out32[i];
        std::cout << out32[i] << " ";
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
