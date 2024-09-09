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

#ifndef ONERT_MICRO_EXECUTE_UTILS_H
#define ONERT_MICRO_EXECUTE_UTILS_H

#include <cmath>
#include "OMStatus.h"
#include "core/reader/OMCircleReader.h"
#include "core/OMRuntimeShape.h"

namespace onert_micro
{
namespace execute
{

void readQuantParams(const circle::Tensor *tensor, long &zero_point, float &scale);
template <typename T>
OMStatus calculateActivationRange(circle::ActivationFunctionType activation, T *activation_min,
                                  T *activation_max)
{
  switch (activation)
  {
    case circle::ActivationFunctionType::ActivationFunctionType_NONE:
      *activation_min = std::numeric_limits<T>::lowest();
      *activation_max = std::numeric_limits<T>::max();
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU:
      *activation_min = 0;
      *activation_max = std::numeric_limits<T>::max();
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU_N1_TO_1:
      *activation_min = -1;
      *activation_max = 1;
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU6:
      *activation_min = 0;
      *activation_max = 6;
      break;
    default:
      assert(false && "Unsupported activation.");
      return UnsupportedActivation;
  }

  return Ok;
}

inline double getQuantizedConvolutionMultipler(float input_scale, float filter_scale,
                                               float output_scale)
{
  const double input_product_scale = static_cast<double>(input_scale * filter_scale);

  assert(input_product_scale >= 0);

  assert(output_scale != 0.f);

  return input_product_scale / static_cast<double>(output_scale);
}

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Handles an arbitrary positive multiplier. The 'shift' output-value is
// basically the 'floating-point exponent' of the multiplier:
// Negative for a right-shift (when the multiplier is <1), positive for a
// left-shift (when the multiplier is >1)
void quantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift);

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of NEGATIVE its exponent ---
// this is intended as a RIGHT-shift.
//
// Restricted to the case where the multiplier < 1 (and non-negative).
void quantizeMultiplierSmallerThanOneExp(double double_multiplier, int32_t *quantized_multiplier,
                                         int *left_shift);

inline std::vector<double>
getQuantizedConvolutionMultiplers(float input_scale, const flatbuffers::Vector<float> *filter_scale,
                                  float output_scale)
{
  std::vector<double> effective_output_scales;
  size_t n = filter_scale->size();
  effective_output_scales.reserve(n);
  for (size_t i = 0; i < n; ++i)
  {
    effective_output_scales.push_back(
      getQuantizedConvolutionMultipler(input_scale, filter_scale->operator[](i), output_scale));
  }
  return effective_output_scales;
}

OMStatus calculateActivationRangeQuantized(circle::ActivationFunctionType activation,
                                           int32_t output_zero_point, float output_scale,
                                           circle::TensorType data_type, int32_t *activation_min,
                                           int32_t *activation_max);

inline int computeOutSize(circle::Padding padding, int image_size, int filter_size, int stride,
                          int dilation_rate = 1)
{
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;

  if (stride == 0)
    return 0;

  switch (padding)
  {
    case circle::Padding_SAME:
      return (image_size + stride - 1) / stride;
    case circle::Padding_VALID:
      return (image_size + stride - effective_filter_size) / stride;
    default:
      return 0;
  }
}

inline int computePadding(int32_t stride, int32_t dilation_rate, int32_t in_size,
                          int32_t filter_size, int32_t out_size)
{
  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  int32_t padding = ((out_size - 1) * stride + effective_filter_size - in_size) / 2;
  return padding > 0 ? padding : 0;
}

inline void computePaddingHeightWidth(int32_t stride_height, int32_t stride_width,
                                      int32_t dilation_rate_height, int32_t dilation_rate_width,
                                      int32_t in_height, int32_t in_width, int32_t filter_height,
                                      int32_t filter_width, circle::Padding padding,
                                      int32_t *padding_h, int32_t *padding_w)
{

  int out_width =
    computeOutSize(padding, in_width, filter_width, stride_width, dilation_rate_width);
  int out_height =
    computeOutSize(padding, in_height, filter_height, stride_height, dilation_rate_height);

  *padding_h =
    computePadding(stride_height, dilation_rate_height, in_height, filter_height, out_height);

  *padding_w = computePadding(stride_width, dilation_rate_width, in_width, filter_width, out_width);
}
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

#include "execute/OMUtils.h"
#include <cmath>

using namespace onert_micro::execute;
using namespace onert_micro;

void onert_micro::execute::quantizeMultiplier(double double_multiplier,
                                              int32_t *quantized_multiplier, int *shift)
{
  if (double_multiplier == 0.0)
  {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }

  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(std::round(q * (int64_t(1) << 31)));

  if (q_fixed == (int64_t(1) << 31))
  {
    q_fixed /= 2;
    ++*shift;
  }
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (*shift < -31)
  {
    *shift = 0;
    q_fixed = 0;
  }
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void onert_micro::execute::quantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                                               int32_t *quantized_multiplier,
                                                               int *left_shift)
{
  assert(double_multiplier < 1.0);
  assert(double_multiplier > 0.0);
  int shift;
  onert_micro::execute::quantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
  assert(shift <= 0);
  *left_shift = shift;
}

namespace
{
OMStatus calculateActivationRangeQuantizedImpl(circle::ActivationFunctionType activation,
                                               int32_t qmin, int32_t qmax, int32_t zero_point,
                                               float scale, int32_t *activation_min,
                                               int32_t *activation_max)
{
  assert(scale != 0.f);

  auto quantize = [scale, zero_point](float x) {
    return zero_point + static_cast<int32_t>(std::round(x / scale));
  };

  switch (activation)
  {
    case circle::ActivationFunctionType::ActivationFunctionType_NONE:
    case circle::ActivationFunctionType::ActivationFunctionType_TANH:
      *activation_min = qmin;
      *activation_max = qmax;
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU:
      *activation_min = std::max(qmin, quantize(0.0f));
      *activation_max = qmax;
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU_N1_TO_1:
      *activation_min = std::max(qmin, quantize(-1.0f));
      *activation_max = std::min(qmax, quantize(1.0f));
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU6:
      *activation_min = std::max(qmin, quantize(0.0f));
      *activation_max = std::min(qmax, quantize(6.0f));
      break;
    default:
      assert(false && "Unsupported activation.");
      return UnsupportedActivation;
  }
  return Ok;
}
} // namespace

OMStatus onert_micro::execute::calculateActivationRangeQuantized(
  circle::ActivationFunctionType activation, int32_t output_zero_point, float output_scale,
  circle::TensorType data_type, int32_t *activation_min, int32_t *activation_max)
{
  int32_t qmin;
  int32_t qmax;
  switch (data_type)
  {
    case circle::TensorType_UINT8:
      qmin = 0;
      qmax = std::numeric_limits<uint8_t>::max();
      break;
    case circle::TensorType_INT8:
      qmin = std::numeric_limits<int8_t>::min();
      qmax = std::numeric_limits<int8_t>::max();
      break;
    case circle::TensorType_INT16:
      // For now, assume that signed int16 type implies signed symmetric quantization.
      assert(output_zero_point == 0);
      qmin = std::numeric_limits<int16_t>::min();
      qmax = std::numeric_limits<int16_t>::max();
      break;
    default:
      assert(false && "Unsupported type.");
      return UnsupportedType;
  }

  return calculateActivationRangeQuantizedImpl(activation, qmin, qmax, output_zero_point,
                                               output_scale, activation_min, activation_max);
}

void onert_micro::execute::readQuantParams(const circle::Tensor *tensor, long &zero_point,
                                           float &scale)
{
  // additional check
  assert(tensor->quantization() != nullptr); // Fix caller
  assert(tensor->quantization()->scale() != nullptr and
         tensor->quantization()->scale()->size() == 1); // Fix caller
  assert(tensor->quantization()->zero_point() != nullptr and
         tensor->quantization()->zero_point()->size() == 1); // Fix caller

  // read zero point
  zero_point = tensor->quantization()->zero_point()->operator[](0);
  // read scale
  scale = tensor->quantization()->scale()->operator[](0);
}

int calculateInputRadius(int input_integer_bits, int input_left_shift, int total_signed_bits)

} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_UTILS_H
