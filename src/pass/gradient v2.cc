/*!
 *  Copyright (c) 2016 by Contributors
 * \file gradients.cc
 * \brief Passes that takes gradient of the graph
 * This code code was modified based on mxnet codebase by Min Lin
 */
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <algorithm>
#include <functional>

namespace nnvm {
namespace pass {
namespace {

// A guard that logs mirroring information for only once.
static bool logged_do_mirror = false;

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

#if BASELINE_BACKWARD_MIRRORING

Graph Gradient(Graph src) {
  if (!logged_do_mirror) {
    if (dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0)) {
      LOG(INFO) << "MXNet has the BASELINE "
                << "mirroring algorithm enabled";
    } else {
      LOG(INFO) << "MXNet has backward mirroring disabled";
    }
    logged_do_mirror = true;
  }

  using nnvm::FGradient;
  using MirrorFun = std::function<bool (const NodePtr&)>;
  using AttrHintFun = std::function<NodeEntry (const NodeEntry& src, const NodeEntry &like)>;

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
  using AggFun = std::function<NodeEntry (std::vector<NodeEntry>&& inputs)>;
  AggFun agg_fun = DefaultAggregateGradient;
  if (src.attrs.count("grad_aggregate_fun") != 0) {
    agg_fun = src.GetAttr<AggFun>("grad_aggregate_fun");
  }
  MirrorFun mirror_fun = nullptr;
  if (src.attrs.count("grad_mirror_fun") != 0) {
    mirror_fun = src.GetAttr<MirrorFun>("grad_mirror_fun");
  }
  AttrHintFun attr_hint_fun = nullptr;
  if (src.attrs.count("attr_hint_fun") != 0) {
    attr_hint_fun = src.GetAttr<AttrHintFun>("attr_hint_fun");
  }
  std::vector<const Op*> zero_ops;
  if (src.attrs.count("zero_ops") != 0) {
    zero_ops = src.GetAttr<std::vector<const Op*> >("zero_ops");
  }
  const Op* copy_op = (src.attrs.count("copy_op") != 0) ?
      Op::Get(src.GetAttr<std::string>("copy_op")) :
      nullptr;

  // topo sort
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

  // construct mirror reduece memory strategy if needed
  std::unordered_map<Node*, NodePtr> mirror_map;
  if (mirror_fun != nullptr) {
    for (const NodePtr& n : topo_order) {
      if (mirror_fun(n)) {
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
      NodePtr fwd_node = (mirror_map.size() == 0 ? ptr : mirror_map.at(ptr.get()));
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

#else  // !BASELINE_BACKWARD_MIRRORING

/// @brief Auxiliary function that builds a backward graph
///          based on the provided mirror function.
Graph _buildBackwardGraph(
    const Graph& src,
    const std::vector<NodeEntry>& xs,
    const std::vector<NodePtr>& topo_order,
    std::unordered_map<Node*, 
      std::vector<GradEntry> > output_grads,
    const std::unordered_map<NodePtr, 
        std::pair<NodePtr, 
                  NodePtr> >& mirror_map);

Graph Gradient(Graph src) {
  if (!logged_do_mirror) {
    if (dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0)) {
      LOG(INFO) << "MXNet has the self-defined "
                << "mirroring algorithm enabled";
    } else {
      LOG(INFO) << "MXNet has backward mirroring disabled";
    }
    logged_do_mirror = true;
  }
  using nnvm::FGradient;

  CHECK_NE(src.attrs.count("grad_ys"), 0U)
      << "Gradient require grad_ys to be presented.";
  CHECK_NE(src.attrs.count("grad_xs"), 0U)
      << "Gradient require grad_xs to be presented.";
  CHECK_NE(src.attrs.count("grad_ys_out_grad"), 0U)
      << "Gradient require grad_ys_out_grad to be presented.";
  const std::vector<NodeEntry>& ys =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys");
  const std::vector<NodeEntry>& xs =
      src.GetAttr<std::vector<NodeEntry> >("grad_xs");
  const std::vector<NodeEntry>& ys_out_grad =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys_out_grad");

  using MirrorFun = std::function<MirrorType(
      const NodePtr&)>;
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

  std::unordered_map<NodePtr,
      std::pair<NodePtr,
                NodePtr> > mirror_map;
  // record the list of mirrored operators,
  // for debugging and logging purpose
  std::unordered_set<std::string> mirror_ops;
  // record the longest mirror path
  std::pair<std::size_t, std::vector<NodePtr> > longest_mirror_path;

  longest_mirror_path.first = 0;

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

  // create a source backward graph, with the mirror map being empty
  nnvm::Graph raw_src_grad = _buildBackwardGraph(src, xs,
          topo_order, output_grads, 
          mirror_map);  // with no manipulation on mirroring
  // record the reference count of each node entry
  // This data structure stores the same information with 
  //   the `ref_count` variable in the `plan_memory` pass.
  const IndexedGraph& raw_src_grad_idx = raw_src_grad.indexed_graph();
  std::vector<uint32_t> raw_src_grad_entry_ref_count
      (raw_src_grad_idx.num_node_entries(), 0);
  static const OpMap<std::function<std::vector<uint32_t> (const NodeAttrs& attrs)> >& 
      fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");

  for (uint32_t nid = 0; nid < raw_src_grad_idx.num_nodes(); ++nid) {
    const IndexedGraph::Node& inode = raw_src_grad_idx[nid];
    if (inode.source->is_variable()) {
      continue;  // variable nodes are ignored
    }
    for (const IndexedGraph::NodeEntry& e : inode.inputs) {
      // increase the entry reference count if it is referenced by an operator input
      ++raw_src_grad_entry_ref_count[raw_src_grad_idx.entry_id(e)];
    }
    if (fignore_inputs.count(inode.source->op()) != 0) {
      std::vector<uint32_t> ignore_inputs = 
          fignore_inputs[inode.source->op()](inode.source->attrs);
      for (const uint32_t i : ignore_inputs) {
        // decrease the entry reference count if it belongs to the ignored inputs
        --raw_src_grad_entry_ref_count[raw_src_grad_idx.entry_id(inode.inputs[i])];
      }
    }  // if (ignore_inputs)
  }  // for (nid ∈ [0, idx.num_nodes())
  for (const IndexedGraph::NodeEntry& e : raw_src_grad_idx.outputs()) {
    // increase the entry reference count if it is set as the entire graph's outputs
    // This is used for prevent the graph outputs 
    //   from being put back to the storage pool.
    ++raw_src_grad_entry_ref_count[raw_src_grad_idx.entry_id(e)];
  }  // for (eid ∈ [0, idx.num_outputs()))

  src.attrs["shape_attr_key"] = 
      std::make_shared<any>(std::string("__shape__"));
  // Should we only inference the shape variable as the storage pass always assume 32-bit?
  src.attrs["dtype_attr_key"] = 
      std::make_shared<any>(std::string("__dtype__"));
  src = ApplyPass(std::move(src), "InferShape");
  src = ApplyPass(std::move(src), "InferType");

  const ShapeVector& src_shape = src.GetAttr<ShapeVector>("shape");

  if (mirror_fun != nullptr) {
    for (const NodePtr& node_ptr : topo_order) {
      std::pair<NodePtr, NodePtr>&
          mirror_node_io_pair =
          mirror_map[node_ptr];
      MirrorType mirror_type = mirror_fun(node_ptr);

      // `mirror_path` stores the path of mirroring, with 
      //   the children nodes always coming before parent
      std::vector<NodePtr> mirror_path;

      if (mirror_type == MirrorType::kNone) {
        mirror_node_io_pair.first  = node_ptr;
        mirror_node_io_pair.second = node_ptr;
      } else {
        NodePtr mirror_node = Node::Create();
        bool all_non_mirror_inputs = true;

        *mirror_node = *node_ptr;
         mirror_node->attrs.name += "_mirror";

        for (NodeEntry& e : mirror_node->inputs) {
          if (e.node != mirror_map.at(e.node).second) {
            all_non_mirror_inputs = false;
          }
          e.node = mirror_map.at(e.node).second;
        }  // for (e ∈ mirror_node->inputs)

        for (NodePtr& n : mirror_node->control_deps) {
          if (n != mirror_map.at(n).second) {
            all_non_mirror_inputs = false;
          }
          n = mirror_map.at(n).second;
        }  // for (d ∈ mirror_node->control_deps)

        if (mirror_type == MirrorType::kInput) {
          mirror_node_io_pair.first  = mirror_node;
          mirror_node_io_pair.second = node_ptr;
        } else {
          // if (mirror_type == MirrorType::kBoth)
          if (all_non_mirror_inputs) {
            const IndexedGraph& idx = src.indexed_graph();
            std::size_t released_storage = 0,
                       allocated_storage = 0;

            for (const NodeEntry& e : node_ptr->inputs) {
              if (raw_src_grad_entry_ref_count[raw_src_grad_idx.entry_id(e)] == 1) {
                // if the reference count of the entry is strictly equal to 1,
                // this implies that the storage allocated for that edge 
                // can be released back to the storage pool

                // Similar to the `plan_memory` pass, we always assume 
                //   tensors to be in 32-bit dtypes.
                released_storage += src_shape[idx.entry_id(e)].Size() * 4;
              }  // if (raw_src_grad_entry_ref_count[entry_id] == 1)
            }  // for (e ∈ src_node->inputs)

            for (uint32_t oidx = 0; oidx < node_ptr->num_outputs(); ++oidx) {
              if (raw_src_grad_entry_ref_count[
                    raw_src_grad_idx.entry_id(
                      raw_src_grad_idx.node_id(node_ptr.get()), oidx
                    )
                  ] == 0) {
                continue;  // ignore outputs that are unused, although it is very unlikely
              }
              allocated_storage += src_shape[idx.entry_id(idx.node_id(node_ptr.get()), oidx)].Size() * 4;
            }  // for (oidx ∈ [0, src_node->num_outputs()))
            if (released_storage >= allocated_storage) {
              mirror_node_io_pair.first  = node_ptr;
              mirror_node_io_pair.second = node_ptr;
              continue;
            }
          }  // if (all_non_mirror_inputs)
          mirror_node_io_pair.first  = mirror_node;
          mirror_node_io_pair.second = mirror_node;
        }  // if (mirror_type == MirrorType::kInput)
      }  // if (mirror_type == MirrorType::kNone)
    }  // for (const NodePtr& node_ptr : topo_order)
  }  // if (mirror_fun != nullptr)

  if (mirror_ops.size() != 0) {
    LOG(INFO) << "You have enabled gradient mirroring.";
    LOG(INFO) << "\t""Given below is "
              << "the list of mirrored operators:";
    for (const std::string &opcode : mirror_ops) {
      LOG(INFO) << "\t\t" << opcode;
    }
  }
  if (longest_mirror_path.first != 0) {
    LOG(INFO) << "\t""Given below is "
              << "the longest mirror path:";
    for (const NodePtr& n : longest_mirror_path.second) {
      LOG(INFO) << "\t\t" << NodePtr2Str(n);
    }
  }

  return _buildBackwardGraph(src, xs,
      topo_order, output_grads,
      mirror_map);
}

Graph _buildBackwardGraph(
    const Graph& src,
    const std::vector<NodeEntry>& xs,
    const std::vector<NodePtr>& topo_order,
    std::unordered_map<Node*, 
      std::vector<GradEntry> > output_grads,
    const std::unordered_map<NodePtr, 
        std::pair<NodePtr,
                  NodePtr> >& mirror_map) {
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
      if (mirror_map.size() != 0) {
        fwd_node = mirror_map.at(ptr).first;
      }

      std::vector<NodeEntry> input_grads;
      if (grad_fun_map.count(ptr->op())) {
        input_grads = grad_fun_map[ptr->op()](fwd_node, out_agg_grads);
        // map the control dependency of the backward 
        // This step is needed to eliminate operators that only require inputs 
        //   for the backward pass (e.g., fully-connected layers).
        // Note that although the outputs of the dead nodes can be immediately
        //   released back to the storage pool, as outputs of layers such as
        //   fully-connected are usually huge, they cannot be easily reused,
        //   resulting in huge increase in overall memory footprint.
        bool is_dead_node = true;

        for (NodeEntry& input_grad_entry : input_grads) {
          for (NodeEntry& input_entry : 
              input_grad_entry.node->inputs) {
            if (input_entry.node == fwd_node) {
              is_dead_node = false;
            }
          }  // for (input_entry ∈ input_grad_entry.node->inputs)
        }  // for (input_grad_entry ∈ input_grads)

        if (dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0) && is_dead_node) {
          for (NodeEntry& input_grad_entry : input_grads) {
            for (NodePtr& control_dep : 
                input_grad_entry.node->control_deps) {
              if (control_dep == fwd_node) {
                control_dep = ptr;

                for (NodePtr& fwd_node_control_dep : 
                    fwd_node->control_deps) {
                  input_grad_entry.node->control_deps.push_back(
                      fwd_node_control_dep);
                }
                break;
              }
            }  // for (control_dep ∈ input_grad.control_deps)
          }  // for (input_grad_entry ∈ input_grads)
        }  // if (is_dead_node)

        // if (dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0) &&
        //     (ptr->attrs.op->name == "BatchNorm" ||
        //      ptr->attrs.op->name == "CuDNNBatchNorm")) {
        //   NodePtr& input_grad_node = input_grads[0].node;
        //   const NodePtr& bn_data_node       = ptr->inputs[0].node;
        //   const uint32_t bn_data_entry_idx  = ptr->inputs[0].index;

        //   for (NodeEntry& input_grad_node_entry : 
        //       input_grad_node->inputs) {
        //     if (input_grad_node_entry.node  == bn_data_node && 
        //         input_grad_node_entry.index == bn_data_entry_idx) {
        //       // find the node entry that corresponds to the data input to the `BatchNorm` layer
        //       NodePtr bn_inv_node = Node::Create();

        //       bn_inv_node->attrs.op = nnvm::Op::Get("BatchNormInv");
        //       bn_inv_node->attrs.name = ptr->attrs.name + "_inv";
        //       bn_inv_node->inputs.push_back(nnvm::NodeEntry{ptr, 0, 0}); // output
        //       bn_inv_node->inputs.push_back(nnvm::NodeEntry{ptr, 1, 0}); // mean
        //       bn_inv_node->inputs.push_back(nnvm::NodeEntry{ptr, 2, 0}); // inv_var
        //       bn_inv_node->inputs.push_back(ptr->inputs[1]); // gamma
        //       bn_inv_node->inputs.push_back(ptr->inputs[2]); // beta
        //       // redirect the input data to the output of the inverse node
        //       input_grad_node_entry = nnvm::NodeEntry{bn_inv_node, 0, 0};
        //     }
        //   }
        //   // LOG(INFO) << "BatchNormGrad Inputs: ";
        //   // for (NodeEntry& input_grad_node_entry : 
        //   //     input_grad_node->inputs) {
        //   //   LOG(INFO) << "\t" << "Node " << input_grad_node_entry.node->attrs.name 
        //   //             << "\t" << "OEdge" << input_grad_node_entry.index;
        //   // }
        // }

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

#endif  // BASELINE_BACKWARD_MIRRORING

// register pass
NNVM_REGISTER_PASS(Gradient)
.describe("Return a gradient graph of src.attrs[\"ys\"] wrt src.attrs[\"xs\"]")
.set_body(Gradient)
.set_change_graph(true)
.depend_graph_attr("shape_inputs")
.depend_graph_attr("dtype_inputs")
.depend_graph_attr("grad_ys")
.depend_graph_attr("grad_xs")
.depend_graph_attr("grad_ys_out_grad");

}  // namespace
}  // namespace pass
}  // namespace nnvm
