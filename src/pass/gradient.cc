/*!
 *  Copyright (c) 2016 by Contributors
 * \file gradients.cc
 * \brief Passes that takes gradient of the graph
 * This code code was modified based on mxnet codebase by Min Lin
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <algorithm>
#include <functional>

namespace nnvm {
namespace pass {
namespace {

// static bool logged_mirror_path = false;

// default aggregate gradient function
// require operator __zero__ and __ewise_sum__ to be presented.
NodeEntry DefaultAggregateGradient(std::vector<NodeEntry>&& v) {
  if (v.size() == 1) {
    return std::move(v[0]);
  } else if (v.size() == 0) {
    NodePtr zero_node = Node::Create();
    zero_node->attrs.op = Op::Get("__zero__");
    return NodeEntry{zero_node, 0, 0};
  } else {
    NodePtr sum_node = Node::Create();
    sum_node->attrs.op = Op::Get("__ewise_sum__");
    sum_node->inputs = std::move(v);
    return NodeEntry{sum_node, 0, 0};
  }
}

bool CheckGradAllZero(const std::vector<NodeEntry>& grads,
                      const std::vector<const Op*>& zero_ops) {
  if (!grads.size() || !zero_ops.size()) return false;
  for (const auto& g : grads) {
    bool found = false;
    for (const auto& op : zero_ops) {
      if (g.node->op() == op) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

// helper entry
struct GradEntry {
#ifdef _MSC_VER
  NodeEntry sum = NodeEntry{nullptr, 0, 0};
#else
  NodeEntry sum{nullptr, 0, 0};
#endif
  std::vector<NodeEntry> grads;
  bool need_attr_hint{true};
};

/// @brief Auxiliary function that builds a backward graph
///          based on the provided mirror function.
Graph _buildBackwardGraph(
    const Graph& src, const std::vector<NodeEntry>& xs,
    const std::vector<NodePtr>& topo_order,
    std::unordered_map<Node*, 
      std::vector<GradEntry> > output_grads,
    const std::unordered_map<NodePtr,
      std::unordered_map<NodePtr, NodePtr> >& mirror_map_modified);

Graph Gradient(Graph src) {
  using nnvm::FGradient;

  CHECK_NE(src.attrs.count("grad_ys"), 0U)
      << "Gradient require grad_ys to be presented.";
  CHECK_NE(src.attrs.count("grad_ys_out_grad"), 0U)
      << "Gradient require grad_ys_out_grad to be presented.";
  CHECK_NE(src.attrs.count("grad_xs"), 0U)
      << "Gradient require grad_xs to be presented.";
  const std::vector<NodeEntry>& ys =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys");
  const std::vector<NodeEntry>& ys_out_grad =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys_out_grad");
  const std::vector<NodeEntry>& xs =
      src.GetAttr<std::vector<NodeEntry> >("grad_xs");

  using MirrorFun = std::function<bool(
      const Node& node,
      const unsigned mirror_depth)>;
  MirrorFun mirror_fun = nullptr;
  if (src.attrs.count("grad_mirror_fun") != 0) {
    mirror_fun = src.GetAttr<MirrorFun>("grad_mirror_fun");
  }

  // initialize topological order and output gradients
  std::vector<NodePtr> topo_order;
  std::unordered_map<Node*, std::vector<GradEntry> > output_grads;

  DFSVisit(ys, [&](const NodePtr& node) {
      if (output_grads.count(node.get()) == 0) {
        output_grads[node.get()].resize(node->num_outputs());
      }
      topo_order.push_back(node);
    });

  CHECK_EQ(ys.size(), ys_out_grad.size());
  for (size_t i = 0; i < ys.size(); ++i) {
    NodeEntry ograd = ys_out_grad[i];
    output_grads[ys[i].node.get()][ys[i].index].grads = { ograd };
  }

  // Check that all xs are reachable from ys
  for (size_t i = 0; i < xs.size(); ++i) {
    CHECK(output_grads.find(xs[i].node.get()) != output_grads.end())
        << "Cannot differentiate with respect to the " << i+1 << "-th variable "
        << "because it is unreachable from the outputs.";
  }
  /*
  // construct mirror reduece memory strategy if needed
  std::unordered_map<Node*, NodePtr> mirror_map;
  if (mirror_fun != nullptr) {
    for (const NodePtr& n : topo_order) {
      if (mirror_fun(*n)) {
        NodePtr new_node = Node::Create();
        *new_node = *n;
        new_node->attrs.name += "_mirror";
        for (auto& e : new_node->inputs) {
          e.node = mirror_map.at(e.node.get());
        }
        for (auto& n : new_node->control_deps) {
          n = mirror_map.at(n.get());
        }
        mirror_map[n.get()] = std::move(new_node);
      } else {
        mirror_map[n.get()] = n;
      }
    }
  }
   */
  // record the reference count of each node entry
  // This data structure stores the same information with 
  //   the `ref_count` variable in the `plan_memory` pass.
  const IndexedGraph& idx = src.indexed_graph();
  std::vector<uint32_t> entry_ref_count
      (idx.num_node_entries(), 0);
  static const OpMap<std::function<std::vector<uint32_t> (const NodeAttrs& attrs)> >& 
      fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const IndexedGraph::Node& inode = idx[nid];
    if (inode.source->is_variable()) {
      continue;
    }
    for (const IndexedGraph::NodeEntry& e : inode.inputs) {
      ++entry_ref_count[idx.entry_id(e)];
    }
    if (fignore_inputs.count(inode.source->op()) != 0) {
      std::vector<uint32_t> ignore_inputs = 
          fignore_inputs[inode.source->op()](inode.source->attrs);
      for (const uint32_t i : ignore_inputs) {
        --entry_ref_count[idx.entry_id(inode.inputs[i])];
      }
    }  // if (ignore_inputs)
  }  // for nid ∈ [0, idx.num_nodes())
  for (const IndexedGraph::NodeEntry& e : idx.outputs()) {
    ++entry_ref_count[idx.entry_id(e)];
  }

  std::unordered_map<NodePtr,
      std::unordered_map<NodePtr, NodePtr>
      > mirror_map_modified;
  // record the list of mirrored operators, for debugging and logging purpose
  std::unordered_set<std::string> mirror_ops;
  // record the statistics on mirror depth
  // std::map<unsigned, unsigned> mirror_depth_stats;

  std::function<std::string(const NodePtr&)> NodePtr2Str =
      [](const NodePtr& ptr) {
    if (ptr->attrs.op == nullptr) {
      // The node is a variable node, only return the attribute name.
      return "(Node=" + ptr->attrs.name + ")";
    } else {
      return "(Node=" + ptr->attrs.name + ", \t" + 
                "Op=" + ptr->attrs.op->name + ")";
    }
  };  // NodePtr2Str

  if (mirror_fun != nullptr) {
    for (const NodePtr& node_ptr : topo_order) {
      std::unordered_map<NodePtr, NodePtr>& mirror_nodes =
          mirror_map_modified[node_ptr];
      std::unordered_set<NodePtr> mirror_boundary;  // boundary of mirror nodes;
      /// @brief  Create a mirror node of the given `NodePtr`.
      /// @param  _node_ptr      node to be considered
      /// @param  mirror_depth   the mirror depth
      ///        (This corresponds to the number of recursive calls made.
      ///         The `mirror_depth` needs to be restricted to limit 
      ///           the performance overhead.
      ///         The optimal maximum, however, still remains to be discovered.)
      /// @return If `_node_ptr` CANNOT pass the mirror function,
      ///           it is directly returned;
      ///         Otherwise `mirror_nodes` is checked first to see whether
      ///           or not a mirror node has been previously
      ///           created mirroring `_node_ptr`;
      ///         Finally a new node is created and inserted into `mirror_nodes`.
      std::function<NodePtr(
           const NodePtr&,
           const NodePtr&,
           const unsigned)> _create_mirror =
          [&mirror_nodes,
           &mirror_boundary,
           &mirror_fun,
           &mirror_ops,
          //  &mirror_depth_stats,
           &node_ptr,
           &NodePtr2Str,
           &_create_mirror]
          (const NodePtr& curr_node_ptr,
           const NodePtr& prev_node_ptr,
           const unsigned mirror_depth) {

            // return directly if the mirror function returns false
            if (!mirror_fun(*curr_node_ptr, mirror_depth)) {
              // record the parent node as one of the node bounaries,
              //   under the condition that it is not `nullptr`
              if (mirror_boundary.find(prev_node_ptr) ==
                  mirror_boundary.end()) {
                mirror_boundary.insert(prev_node_ptr);
              }
              return curr_node_ptr;
            }
            // return the mirrored node
            // if it has already been created before
            std::unordered_map<NodePtr, NodePtr>::iterator mirror_node_iter;
            if ((mirror_node_iter = mirror_nodes.find(curr_node_ptr)) != 
                 mirror_nodes.end()) {
              return mirror_node_iter->second;
            }
            // create a new node and insert it into `mirror_nodes`
            NodePtr new_node = Node::Create();
            *new_node = *curr_node_ptr;
            new_node->attrs.name = curr_node_ptr->attrs.name +
                "_mirror_at_" + node_ptr->attrs.name;
            mirror_ops.insert(new_node->attrs.op->name);

            for (NodeEntry& e : new_node->inputs) {
              e.node = _create_mirror(e.node,
                  curr_node_ptr,
                  mirror_depth + 1);
            }
            for (NodePtr& n : new_node->control_deps) {
              n = _create_mirror(n,
                  curr_node_ptr,
                  mirror_depth + 1);
            }
            return mirror_nodes[curr_node_ptr] = new_node;
          };  // _create_mirror
      _create_mirror(node_ptr, nullptr, 0);

      // start forward propagating from the mirror boundary to upstream nodes
      // If we forward propagate certain computation node, we can potentially
      //   1. release the storage allocated for the inputs
      //   2. reduce  the overhead of mirroring
      // However, at the same time, this also comes with the cost of 
      //   1. allocate extra storage for the outputs
      // Hence, the forward propagation stops when the newly allocated storage 
      //   is strictly greater than the released storage.
      // This requires information on the tensor shape, data type, and entry reference count.
      for (const NodePtr& n : mirror_boundary) {

      }  // for n ∈ mirror_boundary

      // if (!logged_mirror_path) {
      //   if (mirror_nodes.size() != 0) {
      //     LOG(INFO) << "List of Mirrored Nodes @ Node "
      //               << NodePtr2Str(node_ptr);
      //   }
      //   for (const std::pair<NodePtr, NodePtr> &nn_pair
      //       : mirror_nodes) {
      //     LOG(INFO) << "\t" << NodePtr2Str(nn_pair.first);
      //   }
      // }  // if (!logged_mirror_path)
    }  // for (const NodePtr& node_ptr : topo_order)
  }  // if (mirror_fun != nullptr)

  // logged_mirror_path = true;  // mirror path is logged for only once

  if (mirror_ops.size() != 0) {
    LOG(INFO) << "You have enabled gradient mirroring.";
    LOG(INFO) << "\t""Given below is "
              << "the list of mirrored operators:";
    for (const std::string &opcode : mirror_ops) {
      LOG(INFO) << "\t\t" << opcode;
    }
    // LOG(INFO) << "\t""Given below is "
    //           << "the list of mirror depths:";
    // for (const std::pair<unsigned, unsigned> mirror_depth_cnt_pair
    //     : mirror_depth_stats) {
    //   LOG(INFO) << "\t\t" << mirror_depth_cnt_pair.first << " : "
    //                       << mirror_depth_cnt_pair.second;
    // }
  }

  return _buildBackwardGraph(src, xs,
      topo_order, output_grads,
      mirror_map_modified);
}

Graph _buildBackwardGraph(
    const Graph& src, const std::vector<NodeEntry>& xs,
    const std::vector<NodePtr>& topo_order,
    std::unordered_map<Node*, 
      std::vector<GradEntry> > output_grads,
    const std::unordered_map<NodePtr,
      std::unordered_map<NodePtr, NodePtr> >& mirror_map_modified) {
  using AttrHintFun = std::function<NodeEntry(
      const NodeEntry& src,
      const NodeEntry& like)>;
  AttrHintFun attr_hint_fun = nullptr;
  if (src.attrs.count("attr_hint_fun") != 0) {
    attr_hint_fun = src.GetAttr<AttrHintFun>("attr_hint_fun");
  }
  using AggFun = std::function<NodeEntry(std::vector<NodeEntry>&& inputs)>;
  AggFun agg_fun = DefaultAggregateGradient;
  if (src.attrs.count("grad_aggregate_fun") != 0) {
    agg_fun = src.GetAttr<AggFun>("grad_aggregate_fun");
  }
  std::vector<const Op*> zero_ops;
  if (src.attrs.count("zero_ops") != 0) {
    zero_ops = src.GetAttr<std::vector<const Op*> >("zero_ops");
  }
  const Op* copy_op = (src.attrs.count("copy_op") != 0) ?
      Op::Get(src.GetAttr<std::string>("copy_op")) :
      nullptr;

  // traverse backward
  static auto& grad_fun_map = Op::GetAttr<FGradient>("FGradient");
  static auto& finfer_shape = Op::GetAttr<FInferShape>("FInferShape");

  std::vector<NodeEntry> out_agg_grads;
  for (auto rit = topo_order.rbegin(); rit != topo_order.rend(); ++rit) {
    const NodePtr& ptr = *rit;
    if (ptr->is_variable()) continue;
    out_agg_grads.clear();
    auto& out_grad_vec = output_grads.at(ptr.get());
    for (uint32_t i = 0; i < out_grad_vec.size(); ++i) {
      GradEntry& e = out_grad_vec[i];
      e.sum = agg_fun(std::move(e.grads));
      if (e.need_attr_hint && attr_hint_fun != nullptr) {
        e.sum = attr_hint_fun(e.sum, NodeEntry{ptr, 0, i});
      }
      out_agg_grads.push_back(e.sum);
    }
    if ((*rit)->inputs.size() != 0) {
      // NodePtr fwd_node = (mirror_map.size() == 0 ? ptr : mirror_map.at(ptr.get()));
      NodePtr fwd_node = ptr;
      if (mirror_map_modified.size() != 0) {
        const std::unordered_map<NodePtr, NodePtr>& mirror_nodes =
            mirror_map_modified.at(ptr);
        std::unordered_map<NodePtr, NodePtr>::const_iterator mirror_node_iter;
        if ((mirror_node_iter = mirror_nodes.find(ptr))
            != mirror_nodes.end()) {
          fwd_node = mirror_node_iter->second;
        }
      }

      std::vector<NodeEntry> input_grads;
      if (grad_fun_map.count(ptr->op())) {
        input_grads = grad_fun_map[ptr->op()](fwd_node, out_agg_grads);
        CHECK_EQ((*rit)->inputs.size(), input_grads.size())
            << "Gradient function not returning enough gradient";
      } else if (CheckGradAllZero(out_agg_grads, zero_ops)) {
        for (size_t i = 0; i < fwd_node->num_inputs(); ++i) {
          std::ostringstream os;
          if (1 == fwd_node->num_inputs()) {
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
      }
      auto git = input_grads.begin();
      for (auto it = (*rit)->inputs.begin(); it != (*rit)->inputs.end(); ++it, ++git) {
        auto& ge = output_grads[it->node.get()][it->index];
        // if any of the backward op can do shape inference, the hint is not necessary.
        if (finfer_shape.count(git->node->op())) {
          ge.need_attr_hint = false;
        }
        ge.grads.emplace_back(std::move(*git));
      }
    }
  }
  // take out the xs' grads
  Graph ret;
  ret.outputs.resize(xs.size());
  NodeEntryMap<std::pair<size_t, size_t> > unique_grads;
  size_t counter = 0;
  for (const NodeEntry& e : xs) {
    GradEntry& entry = output_grads[e.node.get()][e.index];
    // aggregate sum if there haven't been
    if (entry.sum.node.get() == nullptr) {
      entry.sum = agg_fun(std::move(entry.grads));
      if (entry.need_attr_hint && attr_hint_fun != nullptr) {
        entry.sum = attr_hint_fun(entry.sum, e);
      }
    }
    if (copy_op != nullptr) {
      auto kv = unique_grads.find(entry.sum);
      if (kv == unique_grads.end()) {
        unique_grads.emplace(std::move(entry.sum), std::make_pair(1, counter));
      } else {
        NodePtr copy_node = Node::Create();
        std::ostringstream os;
        os << entry.sum.node->attrs.name << "_" << kv->second.first << "_copy";
        kv->second.first++;
        copy_node->attrs.op = copy_op;
        copy_node->attrs.name = os.str();
        copy_node->inputs.emplace_back(entry.sum);
        if (copy_node->attrs.op->attr_parser != nullptr) {
            copy_node->attrs.op->attr_parser(&(copy_node->attrs));
        }
        unique_grads.emplace(NodeEntry{std::move(copy_node), 0, 0}, std::make_pair(1, counter));
      }
    } else {
        ret.outputs[counter] = entry.sum;
    }
    ++counter;
  }
  if (copy_op != nullptr) {
    for (const auto& kv : unique_grads) {
      ret.outputs[kv.second.second] = kv.first;
    }
  }
  return ret;
}

// register pass
NNVM_REGISTER_PASS(Gradient)
.describe("Return a gradient graph of src.attrs[\"ys\"] wrt src.attrs[\"xs\"]")
.set_body(Gradient)
.set_change_graph(true)
.depend_graph_attr("grad_ys")
.depend_graph_attr("grad_xs")
.depend_graph_attr("grad_ys_out_grad");

}  // namespace
}  // namespace pass
}  // namespace nnvm
