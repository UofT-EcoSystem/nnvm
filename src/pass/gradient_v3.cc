/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2016 by Contributors
 * \file gradients.cc
 * \brief Passes that takes gradient of the graph
 * This code code was modified based on mxnet codebase by Min Lin
 */
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/pass_functions.h>

#include <algorithm>
#include <functional>


namespace nnvm {
namespace pass {
namespace {

struct GradEntry {
#ifdef _MSC_VER
  NodeEntry sum = NodeEntry{nullptr, 0, 0};
#else
  NodeEntry sum{nullptr, 0, 0};
#endif
  std::vector<NodeEntry> grads;
  bool need_attr_hint{true};
};

/// @brief Build a backward graph from the mirroring function.
Graph BuildBackwardGraph(
    const Graph& src,
    const std::vector<NodeEntry>& xs,
    const std::vector<NodePtr>& topo_order,
    std::unordered_map<NodePtr, std::vector<GradEntry> > output_grads,
    const std::unordered_map<NodePtr, std::pair<NodePtr, NodePtr> >& mirror_map);

Graph Gradient(Graph src) {
  using nnvm::FGradient;

  CHECK_NE(src.attrs.count("grad_xs"), 0U)
      << "Gradient require grad_xs to be presented.";
  CHECK_NE(src.attrs.count("grad_ys"), 0U)
      << "Gradient require grad_ys to be presented.";
  CHECK_NE(src.attrs.count("grad_ys_out_grad"), 0U)
      << "Gradient require grad_ys_out_grad to be presented.";
  const std::vector<NodeEntry>& xs =
      src.GetAttr<std::vector<NodeEntry> >("grad_xs");
  const std::vector<NodeEntry>& ys =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys");
  const std::vector<NodeEntry>& ys_out_grad =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys_out_grad");
  CHECK_EQ(ys.size(), ys_out_grad.size());

  using MirrorFun = std::function<MirrorType(const NodePtr&)>;
  MirrorFun mirror_fun = nullptr;
  if (src.attrs.count("mirror_fun") != 0) {
    mirror_fun = src.GetAttr<MirrorFun>("mirror_fun");
  }

  // initialize a topological order of the graph nodes and `output_grads`
  // that maps every operator node to its gradient entries
  std::vector<NodePtr> topo_order;
  std::unordered_map<NodePtr, std::vector<GradEntry> > output_grads;

  DFSVisit(ys, [&](const NodePtr& node) {
        if (output_grads.count(node) == 0) {
          output_grads[node].resize(node->num_outputs());
        }
        topo_order.push_back(node);
      });
  for (size_t i = 0; i < ys.size(); ++i) {
    output_grads[ys[i].node][ys[i].index].grads = {ys_out_grad[i]};
  }

  // sanity check that all xs are reachable from ys
  for (size_t i = 0; i < xs.size(); ++i) {
    CHECK(output_grads.find(xs[i].node.get()) != output_grads.end())
        << "Cannot differentiate with respect to the "
        << i+1 << "-th variable "
        << "because it is unreachable from the outputs.";
  }
}

// Given the list of gradient entries and zero operators, check whether all the
// gradients are zero or not.
bool CheckGradAllZero(const std::vector<NodeEntry>& grads,
                      const std::vector<const Op*>& zero_ops) {
  if (!grads.size() || !zero_ops.size()) {
    return false;
  }
  for (const auto& g : grads) {
    bool is_zero_op = false;
    for (const auto& op : zero_ops) {
      if (g.node->op() == op) {
        is_zero_op = true;
        break;
      }
    }
    if (!is_zero_op) {
      return false;
    }
  }
  return true;
}

Graph BuildBackwardGraph(
    const Graph& src,
    const std::vector<NodeEntry>& xs,
    const std::vector<NodePtr>& topo_order,
    std::unordered_map<NodePtr, std::vector<GradEntry> > output_grads,
    const std::unordered_map<NodePtr, std::pair<NodePtr, NodePtr> >& mirror_map) {
  // gradient aggregation and attribute hint function (The latter is usually set
  // to NULL by the executor frontend)
  using AggregateFun = std::function<NodeEntry(
      std::vector<NodeEntry>&& inputs)>;
  using AttrHintFun = std::function<NodeEntry(
      const NodeEntry& src,
      const NodeEntry& like)>;
  AggregateFun aggregate_fun =
      [](std::vector<NodeEntry>&& v)->NodeEntry {
        if (v.size() == 1) {
          return std::move(v[0]);
        } else if (v.size() == 0) {
          NodePtr zero_grad_node = Node::Create();
          zero_grad_node->attrs.op = Op::Get("__zero__");
          zero_grad_node->attrs.name = "zero_grad";
          return NodeEntry{zero_grad_node, 0, 0};
        } else {
          NodePtr grad_agg_node = Node::Create();
          grad_agg_node->attrs.op = Op::Get("__ewise_sum__");
          grad_agg_node->attrs.name = "grad_sum";
          grad_agg_node->inputs = std::move(v);
          return NodeEntry{grad_agg_node, 0, 0};
        }
      };
  AttrHintFun attr_hint_fun = nullptr;
  if (src.attrs.count("aggregate_fun") != 0) {
    aggregate_fun = src.GetAttr<AggregateFun>("aggregate_fun");
  }
  if (src.attrs.count("attr_hint_fun") != 0) {
    attr_hint_fun = src.GetAttr<AttrHintFun> ("attr_hint_fun");
  }

  // zero and copy operators
  std::vector<const Op*> zero_ops;
  if (src.attrs.count("zero_ops") != 0) {
    zero_ops = src.GetAttr<std::vector<const Op*> >("zero_ops");
  }
  const Op* copy_op = (src.attrs.count("copy_op_str") != 0) ?
      Op::Get(src.GetAttr<std::string>("copy_op_str")) : nullptr;

  static auto& grad_fun_map = Op::GetAttr<FGradient>("FGradient");
  static auto& finfer_shape = Op::GetAttr<FInferShape>("FInferShape");

  std::vector<NodeEntry> out_agg_grads;
  for (auto rit = topo_order.rbegin(); rit != topo_order.rend(); ++rit) {
    const NodePtr& ptr = *rit;
    // skip the current node if it is a variable node
    if (ptr->is_variable()) continue;

    // otherwise gather all the gradient entries and apply the aggregation function
    out_agg_grads.clear();
    std::vector<GradEntry>& out_grad_vec = output_grads.at(ptr);
    for (uint32_t i = 0; i < out_grad_vec.size(); ++i) {
      GradEntry& e = out_grad_vec[i];
      e.sum = aggregate_fun(std::move(e.grads));
      if (e.need_attr_hint && attr_hint_fun != nullptr) {
        e.sum = attr_hint_fun(e.sum, NodeEntry{ptr, 0, i});
      }
      out_agg_grads.push_back(e.sum);
    }  // for (i ∈ out_grad_vec.size())

    if ((*rit)->inputs.size() != 0) {
      // If the current operator node has inputs, we will have to further
      // propagate the gradients backward. 
      NodePtr fwd_node = mirror_map.size() != 0 ? fwd_node = mirror_map.at(ptr).first : ptr;
      
      std::vector<NodeEntry> input_grads;
      if (grad_fun_map.count(ptr->op())) {
        // The gradient function is applied to the forward operator node (or the
        // MIRRORED forward operator node if backward mirroring has been
        // enabled) and the aggregated output gradients.
        input_grads = grad_fun_map[ptr->op()](fwd_node, out_agg_grads);
        if (fwd_node != ptr) {
          // If backward mirroring has been enabled, check whether the mirrored
          // forward node is dead or not. The node is deemed as dead if there
          // is no data dependency between itself and the gradient operator node.
          bool is_fwd_node_dead;
          for (NodeEntry& input_grad_entry : input_grads) {
            for (NodeEntry& input_entry : input_grad_entry.node->inputs) {
              if (input_entry.node == fwd_node) {
                is_fwd_node_dead = false;
              }
            }
          }  // for (input_grad_entry ∈ input_grads)

          if (is_fwd_node_dead) {
            // If the mirrored forward node is dead, we have to replace the
            // control dependency of the mirrored node with the node in the
            // original forward graph, and also push the control dependencies
            // of the mirrored node to the gradient node because the former
            // is about to be removed.
            for (NodeEntry& input_grad_entry : input_grads) {
              for (NodePtr& control_dep : input_grad_entry.node->control_deps) {
                if (control_dep == fwd_node) {
                  control_dep = ptr;
                  for (NodePtr& fwd_node_control_dep : fwd_node->control_deps) {
                    input_grad_entry.node->control_deps.push_back(
                        fwd_node_control_dep);
                  }
                  break;
                }
              }  // for (control_dep ∈ input_grad.control_deps)
            }  // for (input_grad_entry ∈ input_grads)
          }  // if (is_fwd_node_dead)
        }  // if (fwd_node != ptr)
        CHECK_EQ((*rit)->inputs.size(), input_grads.size())
            << "Gradient function not returning enough gradient";
      } else if (CheckGradAllZero(out_agg_grads, zero_ops)) {
        for (size_t i = 0; i < fwd_node->num_inputs(); ++i) {
          std::ostringstream os;
          if (fwd_node->num_inputs() == 1) {
            os << fwd_node->attrs.name << "_backward";
          } else {
            os << fwd_node->attrs.name << "_in" << i << "_backward";
          }
          auto p = Node::Create();
          p->attrs.op = zero_ops[0];
          p->attrs.name = os.str();
          p->inputs.push_back(fwd_node->inputs[i]);
          p->control_deps.emplace_back(fwd_node);
          if (p->op()->attr_parser != nullptr) {
            p->op()->attr_parser(&(p->attrs));
          }
          input_grads.emplace_back(nnvm::NodeEntry{p, 0, 0});
        }
      } else {
        LOG(FATAL) << "Operator " << fwd_node->op()->name << " is non-differentiable "
                   << "because it didn't register FGradient attribute.";
      }  // if (grad_fun_map.count(ptr->op()))
      auto git = input_grads.begin();
      for (auto it = (*rit)->inputs.begin(); it != (*rit)->inputs.end(); ++it, ++git) {
        GradEntry& ge = output_grads[it->node][it->index];
        // If any of the backward op can do shape inference, the attribute hint
        // is not necessary.
        if (finfer_shape.count(git->node->op())) {
          ge.need_attr_hint = false;
        }
        // move the input gradients of the node entries to the output gradients
        // of the next node in reverse topological orders
        ge.grads.emplace_back(std::move(*git));
      }
    }  // if ((*rit)->inputs.size() != 0)
  }  // for (rit ∈ reverse(topo_order))


}

// register pass
NNVM_REGISTER_PASS(MXGradient)
.describe("Return a gradient graph of src.attrs[\"ys\"] wrt src.attrs[\"xs\"]")
.set_body(GradientV3)
.set_change_graph(true)
.depend_graph_attr("grad_ys")
.depend_graph_attr("grad_xs")
.depend_graph_attr("in_arg_shapes")
.depend_graph_attr("in_arg_dtypes")
.depend_graph_attr("grad_ys_out_grad");

}  // namespace anonymous
}  // namespace pass
}  // namespace nnvm
