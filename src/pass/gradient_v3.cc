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
#include <deque>
#include <functional>
#include <queue>
#include <unordered_set>
#include <vector>


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
};

/*!
 * \brief Build a backward graph from the mirroring function.
 */
Graph BuildBackwardGraph(
    const Graph& src,
    const std::vector<NodeEntry>& xs,
    const std::vector<NodePtr>& topo_order,
    std::unordered_map<NodePtr, std::vector<GradEntry> > output_grads,
    const std::unordered_map<const Node*, NodePtr>& mirror_map);


/*!
 * \brief Auxiliary function that checks the data dependency between the forward
 *        node and the gradient node, and will return true if the gradient
 *        dependencies are only on the inputs of the forward node.
 */
inline bool IsGradDepOnlyOnFwdInputs(
    const std::vector<NodeEntry>& input_grads,
    const NodePtr& fwd_node) {
  bool is_grad_dep_only_on_fwd_inputs = false;
  for (const NodeEntry& input_grad_entry : input_grads) {
    for (const NodeEntry& input_grad_input_entry :
         input_grad_entry.node->inputs) {
      if (input_grad_input_entry.node == fwd_node) {
        is_grad_dep_only_on_fwd_inputs = true;
        break;
      }
    }
    if (is_grad_dep_only_on_fwd_inputs) break;
  }
  return is_grad_dep_only_on_fwd_inputs;
}


Graph GradientV3(Graph src) {
  const IndexedGraph& idx = src.indexed_graph();
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

  // initialize a topological order of the graph nodes and `output_grads`
  // that maps every operator node to its gradient entries
  std::vector<NodePtr> topo_order;
  std::unordered_map<NodePtr, std::vector<GradEntry> > output_grads;

  DFSVisit(ys,
           [&](const NodePtr& node) {
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
    CHECK(output_grads.find(xs[i].node) != output_grads.end())
        << "Cannot differentiate with respect to the " 
        << i+1 << "-th variable "
        << "because it is unreachable from the outputs.";
  }

  std::unordered_map<const Node*, NodePtr> mirror_map;

  // complete the backward graph of the src, but without backward mirroring
  nnvm::Graph gsrc_no_mirroring = BuildBackwardGraph(src, xs, topo_order,
                                                     output_grads, mirror_map);
  const IndexedGraph& gsrc_no_mirroring_idx = gsrc_no_mirroring.indexed_graph();

  using MirrorFun = std::function<bool(const Node* const)>;
  MirrorFun mirror_fun = nullptr;
  if (src.attrs.count("mirror_fun") != 0) {
    mirror_fun = src.GetAttr<MirrorFun>("mirror_fun");
  }
  if (mirror_fun == nullptr) {
    return gsrc_no_mirroring;
  }
  // ===========================================================================
  // ----- Gradient Pass w/ Backward Mirroring -----
  // ===========================================================================
  // record, for each node entry ∈ src, the nodes that reference the entry as inputs
  std::vector<std::unordered_set<const Node*> > node_entry_ref_map(
      gsrc_no_mirroring_idx.num_node_entries());
  static const auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
  for (uint32_t nid = 0;
       nid < gsrc_no_mirroring_idx.num_nodes(); ++nid) {
    const IndexedGraph::Node& inode = gsrc_no_mirroring_idx[nid];
    if (inode.source->is_variable()) {
      continue;
    }
    for (uint32_t i = 0; i < inode.inputs.size(); ++i) {
      if (fignore_inputs.count(inode.source->op()) != 0) {
        std::vector<uint32_t> ignore_inputs =
            fignore_inputs[inode.source->op()](inode.source->attrs);
        if (std::find(ignore_inputs.begin(), ignore_inputs.end(), i)
            != ignore_inputs.end()) {
          continue;
        }
      }
      node_entry_ref_map[gsrc_no_mirroring_idx.entry_id(inode.inputs[i])].insert(inode.source);

      if (gsrc_no_mirroring_idx[inode.inputs[i].node_id].source->attrs.name ==
          "decoder_rnn_concat_target_context_t87") {
        LOG(INFO) << "Reference Node: " << inode.source->attrs.name;
      }

    }
  }  // for (nid ∈ gsrc_no_mirroring.num_nodes)

  src.attrs["shape_attr_key"] = std::make_shared<any>(std::string("__shape__"));
  src.attrs["dtype_attr_key"] = std::make_shared<any>(std::string("__dtype__"));
  src = ApplyPass(std::move(src), "InferShape");
  src = ApplyPass(std::move(src), "InferType");

  const ShapeVector& src_shape = src.GetAttr<ShapeVector>("shape");
  // Since the storage allocator always assume 32-bit, the data type vector is
  // not used here.

  static auto& grad_fun_map = Op::GetAttr<FGradient>("FGradient");

  // apply the backward mirroring heuristics
  std::queue<const Node*> worklist;
  // initialize the worklist to the output nodes
  for (const NodeEntry& e : src.outputs) {
    worklist.push(e.node.get());
  }
  for (; !worklist.empty(); worklist.pop()) {
    const Node* const workitem = worklist.front();
    if (workitem->is_variable() ||
        mirror_map.find(workitem) != mirror_map.end()) {
      continue;
    }

    // subgraph and its nodes in topological order
    std::unordered_set<const Node*> subgraph;
    std::deque<const Node*> subgraph_topo_order;
    // The sub-worklist is used for constructing the subgraph. It is initialized
    // to have the current workitem node.
    std::queue<const Node*> subworklist;
    subworklist.push(workitem);
    // Start propagating from the current workitem node backward until the
    // mirroring function returns false (indicating that a compute-heavy layer
    // has been hit), in which case we put the node that fails the mirroring
    // function into the worklist as the new head. During the traversal, we
    // build up the subgraph and its topological order at the same time.

    LOG(INFO) << "Workitem: " << workitem->attrs.name;
    LOG(INFO) << "Subgraph  Size: " << subgraph  .size();    

    auto subworklist_backprop = [&subworklist, &subgraph,
                                 &subgraph_topo_order,
                                 &mirror_fun, &worklist]() {
          std::deque<const Node*> subworklist_topo_order;
          for (; !subworklist.empty(); subworklist.pop()) {
            const Node* const subworkitem = subworklist.front();
            if (subworkitem->is_variable()) continue;
            // if (!mirror_fun(subworkitem)) {
            //   worklist.push(subworkitem);
            //   continue;
            // }
            if (subgraph.find(subworkitem) == subgraph.end()) {
              subgraph.insert(subworkitem);
              subworklist_topo_order.push_front(subworkitem);
            }
            for (const NodeEntry& e : subworkitem->inputs) {
              if (!mirror_fun(e.node.get())) {
                worklist.push(e.node.get());
              } else {
                subworklist.push(e.node.get());
              }
            }
            for (const NodePtr& n : subworkitem->control_deps) {
              if (!mirror_fun(n.get())) {
                worklist.push(n.get());
              } else {
                subworklist.push(n.get());
              }
            }
          }  // while (!subworklist.empty())
          subgraph_topo_order.insert(subgraph_topo_order.end(),
                                     subworklist_topo_order.begin(),
                                     subworklist_topo_order.end());
        };
    subworklist_backprop();
    LOG(INFO) << "Backward Pass Ends Here";

    LOG(INFO) << "Subgraph  Size: " << subgraph  .size();

    // =========================================================================
    // ----- Backward Pass Ends Here -----
    // =========================================================================
    bool has_subgraph_converged = false;
    while (!has_subgraph_converged) {
      has_subgraph_converged = true;
      for (const Node* subgraph_node : subgraph_topo_order) {

        // LOG(INFO) << "Processing Subgraph Node: " << subgraph_node->attrs.name;

        for (const NodeEntry& subgraph_node_entry :
             subgraph_node->inputs) {
          const std::unordered_set<const Node*>& ref_nodes =
              node_entry_ref_map[gsrc_no_mirroring_idx.entry_id(subgraph_node_entry)];

          // std::cout << "Reference Nodes: ";
          // for (const Node* n : ref_nodes) {
          //   std::cout << n->attrs.name << " -> ";
          // }
          // std::cout << std::endl;

          // if there are other nodes that reference the node entry and that
          // node satisfies the following condition:
          //   (1) belongs to the forward graph
          //   (2) is not part of the subgraph
          //   (3) passes the mirroring function
          // we add that node to the subgraph and adjust the topological order
          for (const Node* ref_node : ref_nodes) {

            // LOG(INFO) << "Processing Reference Node: " << ref_node->attrs.name;

            if (ref_node != subgraph_node && idx.exist(ref_node) &&
                subgraph.find(ref_node) == subgraph.end() &&
                mirror_fun(ref_node)) {
              // forward propagate from the reference node until the mirroring
              // function returns false
              std::queue<const Node*> ref_node_heads;
              ref_node_heads.push(ref_node);
              for (; !ref_node_heads.empty(); ref_node_heads.pop()) {
                const Node* ref_node_head = ref_node_heads.front();

                // LOG(INFO) << "Processing Reference Node Head: " << ref_node_head->attrs.name;

                if (!mirror_fun(ref_node_head)) {

                  LOG(INFO) << "Subgraph Node: " << subgraph_node->attrs.name;
                  LOG(INFO) << "New Subworklist Head: " << ref_node_head->attrs.name; 

                  subworklist.push(ref_node_head);
                  continue;
                }
                uint32_t nid = gsrc_no_mirroring_idx.node_id(ref_node_head);
                for (uint32_t oid = 0; oid < ref_node_head->num_outputs(); ++oid) {
                  uint32_t eid = gsrc_no_mirroring_idx.entry_id(nid, oid);

                  // std::cout << "Reference Nodes in FwdProp: ";
                  // for (const Node* n : node_entry_ref_map[eid]) {
                  //   std::cout << n->attrs.name << " -> ";
                  // }
                  // std::cout << std::endl;

                  for (const Node* const n : node_entry_ref_map[eid]) {
                    if (idx.exist(n)) {

                      LOG(INFO) << "Pushing " << n->attrs.name << " to the ref_node_heads";

                      ref_node_heads.push(n);
                    }
                  }
                }  // for (oid ∈ [0, ref_node_head->num_outputs()))
              }  // while (!ref_node_heads.empty())

              // LOG(INFO) << "Subgraph in Topo-Order: ";
              // for (const Node* n : subgraph_topo_order)
              //   std::cout << n->attrs.name << " (" << n->op()->name << ")" << " -> ";
              // std::cout << std::endl;
              // std::cout << "Subgraph Size: " << subgraph.size() << std::endl;

              subworklist_backprop();

              // LOG(INFO) << "Subgraph in Topo-Order: ";
              // for (const Node* n : subgraph_topo_order)
              //   std::cout << n->attrs.name << " (" << n->op()->name << ")" << " -> ";
              // std::cout << std::endl;
              // std::cout << "Subgraph Size: " << subgraph.size() << std::endl;

              // We can safely insert the current node at the end of the list
              // WITHOUT violating the topological order, the reason is because
              // since the node has never been inserted before to the subgraph,
              // neither should its outputs (otherwise it violates the property
              // of the backward pass).
              has_subgraph_converged = false;
              break;
            }
          }  // for (ref_node ∈ ref_nodes)
          if (!has_subgraph_converged) {
            break;
          }
        }  // for (subgraph_node_entry ∈ subgraph_node->inputs)
        if (!has_subgraph_converged) {
          break;
        }
      }  // for (subgraph_node ∈ subgraph_topo_order)
    }  // while (!has_subgraph_converged)
    LOG(INFO) << "MirrorMap Size: " << mirror_map.size();
    LOG(INFO) << "Subgraph  Size: " << subgraph  .size();
    LOG(INFO) << "Subgraph in Topo-Order: ";
    for (const Node* n : subgraph_topo_order)
      std::cout << n->attrs.name << " (" << n->op()->name << ")" << " -> ";
    std::cout << std::endl;
    LOG(INFO) << "Subgraph Construction Ends Here";
    // =========================================================================
    // ----- Subgraph Construction Ends Here -----
    // =========================================================================
    std::unordered_map<uint32_t, uint32_t> subgraph_node_entry_ref_cnt;
    for (const Node* subgraph_node : subgraph_topo_order) {
      CHECK(idx.exist(subgraph_node)) << "Every subgraph node must be part of the forward graph.";
      uint32_t nid = gsrc_no_mirroring_idx.node_id(subgraph_node);
      for (uint32_t oid = 0; oid < subgraph_node->num_outputs(); ++oid) {
        uint32_t eid = gsrc_no_mirroring_idx.entry_id(nid, oid);
        subgraph_node_entry_ref_cnt[eid] = node_entry_ref_map[eid].size();
      }
    }  // for (subgraph_node ∈ subgraph_nodes)
    for (const Node* subgraph_node : subgraph_topo_order) {
      mirror_map[subgraph_node] = nullptr;
      if (mirror_fun(subgraph_node)) {
        // if the node satisfies the mirroring function, we compare the memory
        // allocated vs. the memory released by forward propagating the node
        uint32_t newly_allocated_memory = 0, released_memory = 0,
                 nid = gsrc_no_mirroring_idx.node_id(subgraph_node);
        for (uint32_t oid = 0; oid < subgraph_node->num_outputs(); ++oid) {
          uint32_t eid = gsrc_no_mirroring_idx.entry_id(nid, oid);
          newly_allocated_memory += src_shape[eid].Size() * sizeof(float);
        }
        for (const NodeEntry& e : subgraph_node->inputs) {
          uint32_t eid = gsrc_no_mirroring_idx.entry_id(e),
                   ref_cnt = subgraph_node_entry_ref_cnt[eid];
          --ref_cnt;
          if (ref_cnt == 0) {
            released_memory += src_shape[eid].Size() * sizeof(float);
          }
        }
        if (released_memory > newly_allocated_memory) {
          // mark node as to be mirrored
          NodePtr subgraph_node_mirror = Node::Create();
          *subgraph_node_mirror = *subgraph_node;
          subgraph_node_mirror->attrs.name += "_mirror";
          std::unordered_map<const Node*, NodePtr> ::iterator mirror_map_iter;
          for (NodeEntry& e : subgraph_node_mirror->inputs) {
            mirror_map_iter = mirror_map.find(e.node.get());
            // e.node = mirror_map_iter == mirror_map.end() || mirror_map_iter->second == nullptr ?
            //          e.node : mirror_map_iter->second;
            e.node = mirror_map_iter == mirror_map.end() ? e.node : mirror_map_iter->second;
          }
          for (NodePtr& n : subgraph_node_mirror->control_deps) {
            mirror_map_iter = mirror_map.find(n.get());
            // n = mirror_map_iter == mirror_map.end() || mirror_map_iter->second == nullptr ? n : mirror_map_iter->second;
            n = mirror_map_iter == mirror_map.end() ? n : mirror_map_iter->second;
          }
          mirror_map[subgraph_node] = subgraph_node_mirror;
        }  // if (released_memory > newly_allocated_memory)
      } else {
        // If the subgraph node fails the mirorring condition, it is however
        // still possible for it to be mirrored under the condition that its
        // corresponding gradient node only has data dependencies on its inputs.
        NodePtr fake_out_grad_node = Node::Create();
        *fake_out_grad_node = *subgraph_node;
        if (grad_fun_map.count(subgraph_node->op())) {
          std::vector<NodeEntry> fake_out_grads;
          for (uint32_t oid = 0; oid < fake_out_grad_node->num_outputs(); ++oid) {
            fake_out_grads.push_back(NodeEntry{fake_out_grad_node, oid, 0});
          }
          std::vector<NodeEntry> input_grads =
              grad_fun_map[subgraph_node->op()](fake_out_grad_node, fake_out_grads);
          if (IsGradDepOnlyOnFwdInputs(input_grads, fake_out_grad_node)) {
            mirror_map[subgraph_node] = fake_out_grad_node;
          }
        }
      }  // if (mirror_fun(subgraph_node))
    }  // for (subgraph_node ∈ subgraph_topo_order)
    LOG(INFO) << "Forward Pass Ends Here";
    // =========================================================================
    // ----- Forward Pass Ends Here -----
    // =========================================================================
  }  // while (!worklist.empty)
  LOG(INFO) << "Finished the Echo compiler pass";
  LOG(INFO) << "MirrorMap Size: " << mirror_map.size() << " vs. "
            << "Number of Operator Nodes: " << idx.num_nodes();
  DFSVisit(ys,
           [&](const NodePtr& node) {
             if (mirror_map[node.get()] != nullptr) {
               if (mirror_fun(node.get())) {
                 node->attrs.dict["__mirror_stage__"] = "2";
               } else {
                 node->attrs.dict["__mirror_stage__"] = "1";
               }
             } else {
               node->attrs.dict["__mirror_stage__"] = "0";
             }
           });
  return BuildBackwardGraph(src, xs, topo_order, output_grads, mirror_map);
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
    const std::unordered_map<const Node*, NodePtr>& mirror_map) {
  // gradient aggregation and attribute hint function (The latter is usually set
  // to NULL by the executor frontend)
  using AggregateFun = std::function<NodeEntry(
      std::vector<NodeEntry>&& inputs)>;
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
  if (src.attrs.count("aggregate_fun") != 0) {
    aggregate_fun = src.GetAttr<AggregateFun>("aggregate_fun");
  }

  // zero and copy operators
  std::vector<const Op*> zero_ops;
  if (src.attrs.count("zero_ops") != 0) {
    zero_ops = src.GetAttr<std::vector<const Op*> >("zero_ops");
  }
  const Op* copy_op = (src.attrs.count("copy_op_str") != 0) ?
      Op::Get(src.GetAttr<std::string>("copy_op_str")) : nullptr;

  static auto& grad_fun_map = Op::GetAttr<FGradient>("FGradient");

  std::vector<NodeEntry> out_agg_grads;
  for (auto rit = topo_order.rbegin(); rit != topo_order.rend(); ++rit) {
    const NodePtr& ptr = *rit;
    // skip the current node if it is a variable node
    if (ptr->is_variable()) continue;

    // otherwise gather all the gradient entries and apply the aggregation function
    out_agg_grads.clear();
    std::vector<GradEntry>& out_grad_vec = output_grads.at(ptr);
    for (uint32_t i = 0; i < out_grad_vec.size(); ++i) {
      GradEntry& grad_entry = out_grad_vec[i];
      grad_entry.sum = aggregate_fun(std::move(grad_entry.grads));
      out_agg_grads.push_back(grad_entry.sum);
    }  // for (i ∈ out_grad_vec.size())

    if ((*rit)->inputs.size() != 0) {
      // If the current operator node has inputs, we will have to further
      // propagate the gradients backward.
      NodePtr fwd_node = (mirror_map.empty() ||
                          mirror_map.at(ptr.get()) == nullptr) ?
                         ptr : mirror_map.at(ptr.get());
      // NodePtr fwd_node = ptr;
      std::vector<NodeEntry> input_grads;
      if (grad_fun_map.count(ptr->op())) {
        // The gradient function is applied to the forward operator node (or the
        // MIRRORED forward operator node if backward mirroring has been
        // enabled) and the aggregated output gradients.
        input_grads = grad_fun_map[ptr->op()](fwd_node, out_agg_grads);
        if (fwd_node != ptr && IsGradDepOnlyOnFwdInputs(input_grads, fwd_node)) {
          // If the mirrored forward node is dead, we have to replace the
          // control dependency of the mirrored node with the node in the
          // original forward graph.
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
            for (NodePtr& fwd_node_control_dep : fwd_node->control_deps) {
              input_grad_entry.node->control_deps.push_back(
                  fwd_node_control_dep);
            }
          }  // for (input_grad_entry ∈ input_grads)
        }  // if (fwd_node != ptr && IsGradDepOnlyOnFwdInputs)
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
        // move the input gradients of the node entries to the output gradients
        // of the next node in reverse topological orders
        output_grads[it->node][it->index].grads.emplace_back(std::move(*git));
      }  // for (it ∈ rit->inputs, git ∈ input_grads)
    }  // if ((*rit)->inputs.size() != 0)
  }  // for (rit ∈ reverse(topo_order))

  // take out the xs' grads
  Graph ret;
  ret.outputs.resize(xs.size());
  NodeEntryMap<std::pair<size_t, size_t> > unique_grads;
  size_t counter = 0;
  for (const NodeEntry& x : xs) {
    GradEntry& xgrad_entry = output_grads[x.node][x.index];
    // aggregate the gradients for every x, similar to what we did previously
    if (xgrad_entry.sum.node.get() == nullptr) {
      xgrad_entry.sum = aggregate_fun(std::move(xgrad_entry.grads));
    }
    if (copy_op != nullptr) {
      auto kv = unique_grads.find(xgrad_entry.sum);
      if (kv == unique_grads.end()) {
        unique_grads.emplace(std::move(xgrad_entry.sum), std::make_pair(1, counter));
      } else {
        NodePtr copy_node = Node::Create();
        std::ostringstream os;
        os << xgrad_entry.sum.node->attrs.name << "_" << kv->second.first << "_copy";
        kv->second.first++;
        copy_node->attrs.op = copy_op;
        copy_node->attrs.name = os.str();
        copy_node->inputs.emplace_back(xgrad_entry.sum);
        if (copy_node->attrs.op->attr_parser != nullptr) {
          copy_node->attrs.op
                   ->attr_parser(&(copy_node->attrs));
        }
        unique_grads.emplace(NodeEntry{std::move(copy_node), 0, 0}, std::make_pair(1, counter));
      }
    } else {
      ret.outputs[counter] = xgrad_entry.sum;
    }
    ++counter;
  }  // for (e ∈ xs)
  if (copy_op != nullptr) {
    for (const auto& kv : unique_grads) {
      ret.outputs[kv.second.second] = kv.first;
    }
  }
  return ret;
}

// register pass
NNVM_REGISTER_PASS(GradientV3)
.describe("Return a gradient graph of src.attrs[\"ys\"] wrt src.attrs[\"xs\"]")
.set_body(GradientV3)
.set_change_graph(true)
.depend_graph_attr("grad_ys")
.depend_graph_attr("grad_xs")
.depend_graph_attr("shape_inputs")
.depend_graph_attr("dtype_inputs")
.depend_graph_attr("grad_ys_out_grad");

}  // namespace anonymous
}  // namespace pass
}  // namespace nnvm
