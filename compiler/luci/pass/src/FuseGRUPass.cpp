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

#include "luci/Pass/FuseGRUPass.h"
#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>

#include <cmath>

#include <cassert>

// Helper to fuse GRU
namespace
{

class GRUPattern final
{
public:
  GRUPattern(luci::CircleWhileOut *candidate)
  {
    assert(candidate);
    _while_out_node = candidate;
  }
  ~GRUPattern() = default;

  bool matched();

public:
  luci::CircleNode *_ifm = nullptr;
  luci::CircleConst *_weight_ih = nullptr;
  luci::CircleConst *_bias_ih = nullptr;
  luci::CircleConst *_weight_hh = nullptr;
  luci::CircleConst *_bias_hh = nullptr;

  luci::CircleConst *_hidden_input = nullptr;

  luci::CircleConst *_less_const = nullptr;

  luci::CircleWhile *_while_node = nullptr;
  luci::CircleWhileOut *_while_out_node = nullptr;

  luci::CircleReshape *reshape = nullptr;
  luci::CircleConst *reshape_shape = nullptr;

  luci::CircleAdd *add_6 = nullptr;
  luci::CircleMul *mul_1 = nullptr;
  luci::CircleMul *mul_3 = nullptr;
  luci::CircleSub *sub_with_const = nullptr;
  luci::CircleTanh *tanh = nullptr;
  luci::CircleLogistic *logistic_2 = nullptr;
  luci::CircleAdd *add_5 = nullptr;
  luci::CircleMul *mul_2 = nullptr;
  luci::CircleAdd *add_1 = nullptr;
  luci::CircleSplitOut *split_1_out = nullptr;
  luci::CircleSplitOut *split_2_out = nullptr;
  luci::CircleSplit *split_1 = nullptr;
  luci::CircleSplit *split_2 = nullptr;
  luci::CircleLogistic *logistic_1 = nullptr;
  luci::CircleAdd *add_4 = nullptr;
  luci::CircleFullyConnected *fc_1 = nullptr;
  luci::CircleFullyConnected *fc_2 = nullptr;
};

/**
 * Below diagram shows GRU pattern to fuse.
 * Note: this pattern for GRU with `return_sequences=False`
 * - the below pattern will be replaced with one GRU
 *  Main Graph:
 *           [In]    [CircleConst]    [CircleConst]    [CircleConst]    [CircleConst]
 *            |             |              |                |                 |
 *            V             |              |                |                 |
 *         [CircleWhile]<-----------------------------------------------------
 *            |
 *            V
 *      [CircleWhileOut]
 *            |
 *            V
 *          [Out]
 *
 * Condition Graph:
 *          [In]   [CircleConst] (scalar int32 value)
 *           |        |
 *           V        |
 *         [Less]------
 *           |
 *           V
 *         [Out]
 *
 *  Body Graph must contain:
 *  - 2 CircleFullyConnected nodes;
 *  - 3 CircleMul nodes;
 *  - 2 CircleLogistic nodes;
 *  - 2 CircleSplit nodes;
 *  - 6 CircleAdd nodes;
 *  - 1 CircleGather node;
 *  - 1 CircleReshape node;
 *  - 1 CircleSub node;
 *  - 1 CircleTanh node;
 *  - 6 CircleSplitOut nodes;
 *  - 5 CircleInput nodes;
 *  - 5 CircleOutput nodes;
 *
 *  Body Graph:
 *  [In_1]                     [In_2]--->[Add_2 (with Const)]--->[Out_2]            [In_3]
 *    |    \                                                      |                   |
 *    |     \                                         [In_4]---[Gather]      [Add_3 (with Const)]
 *    |     [FullyConnected_1]                          |         |                   |
 *    |               |                              [Out_4]      |                [Out_3]
 *    |           [Split_1]                                 [FullyConnected_2]
 *    |         /         |   \                                       |
 *    |        |          |    \                                   [Split_2]
 *    |    [Add_1] ----------------------------------------------/    |     |
 *    |         |         |    |                                      |     |
 *    |         |         |    ------------------------------------[Add_4]  |
 *    |         |         |                                           |     |
 *    |         |         |                                   [Logistic_1]  |
 *    |         |         |                                          |      |
 *    |         |         ----------------------------------------[Mul_2]   |
 *    |         |                                                    \      /
 *    |         |                                                     [Add_5]
 *    |         |                                                        |
 *    |      [Logistic_2]                                              [Tanh]
 *     \     /           \                                               |
 *      [Mul_1]          [Sub (with const)]                              |
 *       \                                 \                             |
 *        \                                 ---------------------------[Mul_3]
 *         \                                                           /
 *          \                                                         /
 *           --------------------[Add_6]------------------------------
 *                              /     \
 *                             /       \
 *                         [Reshape]   [Out_5]
 *                             |
 *                           [Out_1]
 */

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

bool GRUPattern::matched() { return false; }

class FuseGRU final
{
public:
  FuseGRU(const GRUPattern *p) : _p(p) {}

public:
  void apply(void);

private:
  luci::CircleGRU *create_circle_gru(loco::Graph *graph);

private:
  const GRUPattern *_p;
};

template <loco::DataType T>
void copy_values(const luci::CircleConst *node, luci::CircleConst *cloned)
{
  assert(T == node->dtype());
  assert(T == cloned->dtype());

  const auto size = node->size<T>();
  cloned->size<T>(size);
  for (uint32_t i = 0; i < size; i++)
    cloned->at<T>(i) = node->at<T>(i);
}

luci::CircleConst *clone_circleconst(luci::CircleConst *node, loco::Graph *graph)
{
  auto cloned = graph->nodes()->create<luci::CircleConst>();

  if (cloned != nullptr)
  {
    // dtype/shape
    cloned->dtype(node->dtype());
    cloned->rank(node->rank());

    // values
    switch (node->dtype())
    {
      case loco::DataType::FLOAT32:
        copy_values<loco::DataType::FLOAT32>(node, cloned);
        break;

      case loco::DataType::U8:
        copy_values<loco::DataType::U8>(node, cloned);
        break;

      case loco::DataType::S8:
        copy_values<loco::DataType::S8>(node, cloned);
        break;

      case loco::DataType::S16:
        copy_values<loco::DataType::S16>(node, cloned);
        break;

      case loco::DataType::S32:
        copy_values<loco::DataType::S32>(node, cloned);
        break;

      case loco::DataType::S64:
        copy_values<loco::DataType::S64>(node, cloned);
        break;

      case loco::DataType::BOOL:
        copy_values<loco::DataType::BOOL>(node, cloned);
        break;

      default:
        throw std::runtime_error("FuseGRU: Unsupported data type");
    }
  }

  return cloned;
}

luci::CircleGRU *FuseGRU::create_circle_gru(loco::Graph *graph)
{
  assert(graph);

  auto weight_ih_cloned = clone_circleconst(_p->_weight_ih, graph);
  luci::copy_common_attributes(_p->_weight_ih, weight_ih_cloned);

  auto weight_hh_cloned = clone_circleconst(_p->_weight_hh, graph);
  luci::copy_common_attributes(_p->_weight_hh, weight_hh_cloned);

  luci::CircleNode *bias_ih_cloned = nullptr;
  if (_p->_bias_ih != nullptr)
  {
    bias_ih_cloned = clone_circleconst(_p->_bias_ih, graph);
    luci::copy_common_attributes(_p->_bias_ih, bias_ih_cloned);
  }
  else
  {
    bias_ih_cloned = graph->nodes()->create<luci::CircleOutputExclude>();
  }

  luci::CircleNode *bias_hh_cloned = nullptr;
  if (_p->_bias_hh != nullptr)
  {
    bias_hh_cloned = clone_circleconst(_p->_bias_hh, graph);
    luci::copy_common_attributes(_p->_bias_hh, bias_hh_cloned);
  }
  else
  {
    bias_hh_cloned = graph->nodes()->create<luci::CircleOutputExclude>();
  }

  auto hidden_input_cloned = clone_circleconst(_p->_hidden_input, graph);
  luci::copy_common_attributes(_p->_hidden_input, hidden_input_cloned);

  auto less_const_cloned = clone_circleconst(_p->_less_const, graph);
  luci::copy_common_attributes(_p->_less_const, less_const_cloned);

  // Create and configure new CircleGRU operation.
  auto circle_gru = graph->nodes()->create<luci::CircleGRU>();
  circle_gru->input(_p->_ifm);
  circle_gru->hidden_hidden(weight_hh_cloned);
  circle_gru->hidden_input(weight_ih_cloned);
  circle_gru->hidden_hidden_bias(bias_hh_cloned);
  circle_gru->hidden_input_bias(bias_ih_cloned);
  circle_gru->state(hidden_input_cloned);

  // Note: Now support only returnSequences = false
  circle_gru->returnSequences(false);
  circle_gru->name(_p->_while_node->name() + "_FusedCircleGRU");

  return circle_gru;
}

void FuseGRU::apply()
{
  auto graph = _p->_while_out_node->graph();

  auto gru_out = create_circle_gru(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p->_while_node), luci::get_origin(_p->_while_out_node),
    luci::get_origin(_p->_weight_hh), luci::get_origin(_p->_weight_ih)};

  luci::add_origin(gru_out, luci::composite_origin(origin_vec));

  replace(_p->_while_out_node).with(gru_out);
}

} // namespace

namespace
{

bool fuse_gru(luci::CircleWhileOut *while_out_node)
{
  assert(while_out_node);

  // check first pattern
  GRUPattern pattern(while_out_node);
  if (pattern.matched())
  {
    FuseGRU fuse(&pattern);
    fuse.apply();
    return true;
  }

  return false;
}

} // namespace

namespace luci
{

bool FuseGRUPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto while_out_node = dynamic_cast<luci::CircleWhileOut *>(node);
    if (not while_out_node)
      continue;

    if (fuse_gru(while_out_node))
      changed = true;
  }

  return changed;
}

} // namespace luci
