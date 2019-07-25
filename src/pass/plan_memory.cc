/*!
 *  Copyright (c) 2016 by Contributors
 * \file plan_memory.cc
 * \brief Assign memory tag to each of the data entries.
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <memory>
#include "./graph_algorithm.h"

namespace nnvm {
namespace pass {
namespace {

static bool logged_use_coalesce_softmax = false;

// simple graph based allocator.
class GraphAllocator {
 public:
  // storage id equals integer.
  using StorageID = int;

  // bad storage id
  static const StorageID kBadStorageID = -1;
  // external storage id
  static const StorageID kExternalStorageID = -2;
  // dynamic storage id
  static const StorageID kDynamicStorageID = -3;

  // request a free storage
  StorageID Request(int dev_id, int dtype, TShape shape, uint32_t node_id) {
    if (shape.ndim() == 0) return kBadStorageID;
    // search memory block in [size / match_range_, size * match_range_)
    // TODO(tqchen) add size of the dtype, assume 4 bytes for now
    size_t size = shape.Size() * 4;
    if (match_range_ == 0) return this->Alloc(dev_id, size);
    auto begin = free_.lower_bound(size / match_range_);
    auto mid = free_.lower_bound(size);
    auto end = free_.upper_bound(size * match_range_);
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != node_color_[node_id]) continue;

      // if ((*idx_)[e->released_by_node].source->attrs.name
      //     == "decoder_rnn_l0_t6_i2h") {
      //   std::cout << "The storage released by decoder_rnn_l0_t6_i2h is taken by node " 
      //             << (*idx_)[node_id].source->attrs.name << std::endl;
      // }
      // if ((*idx_)[node_id].source->attrs.name 
      //     == "decoder_rnn_l0_t6_i2h") {
      //   std::cout << "Node " << (*idx_)[node_id].source->attrs.name << " "
      //             << "is taking the storage released by Node "
      //             << (*idx_)[e->released_by_node].source->attrs.name << std::endl;
      // }

      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      // find a exact match, erase from map and return
      free_.erase(it);
      return e->id;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != node_color_[node_id]) continue;

      // if ((*idx_)[e->released_by_node].source->attrs.name
      //     == "decoder_rnn_l0_t6_i2h") {
      //   std::cout << "The storage released by decoder_rnn_l0_t6_i2h is taken by node " 
      //             << (*idx_)[node_id].source->attrs.name << std::endl;
      // } 
      // if ((*idx_)[node_id].source->attrs.name 
      //     == "decoder_rnn_l0_t6_i2h") {
      //   std::cout << "Node " << (*idx_)[node_id].source->attrs.name << " "
      //             << "is taking the storage released by Node "
      //             << (*idx_)[e->released_by_node].source->attrs.name << std::endl;
      // }

      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      // erase from map and return
      free_.erase(it);
      return e->id;
    }
    // cannot find anything return a new one.
    return this->Alloc(dev_id, size);
  }
  // release a memory space.
  void Release(StorageID id, uint32_t node_id) {
    CHECK_NE(id, kBadStorageID);
    if (id == kExternalStorageID || id == kDynamicStorageID) return;
    StorageEntry *e = data_[id].get();
    e->released_by_node = node_id;
    free_.insert({e->max_bytes, e});
  }

  // totoal number of bytes allocated
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (auto &p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  // constructor
  explicit GraphAllocator(const IndexedGraph* idx, const size_t match_range) : idx_(idx) {
    this->Init(match_range, dmlc::GetEnv("NNVM_EXEC_NUM_TEMP", 1));
  }

 private:
  // initialize the graph allocator
  void Init(const size_t match_range, const uint32_t num_match_color) {
    match_range_ = match_range;
    num_match_color_ = num_match_color;
    if (num_match_color_ > 1) {
      std::vector<uint32_t> importance(idx_->num_nodes(), 0);
      for (uint32_t nid = 0; nid < idx_->num_nodes(); ++nid) {
        if ((*idx_)[nid].source->is_variable()) continue;
        importance[nid] = 1;
      }
      num_match_color_ = pass::ColorNodeGroup(
          *idx_, importance, num_match_color_, &node_color_);
    }
  }

  StorageID Alloc(int dev_id, size_t size) {
    StorageID id = static_cast<StorageID>(data_.size());
    std::unique_ptr<StorageEntry> ptr(new StorageEntry());
    ptr->id = id;
    ptr->device_id = dev_id;
    ptr->max_bytes = size;
    data_.emplace_back(std::move(ptr));
    return id;
  }
  // internal storage entry
  struct StorageEntry {
    // the id of the entry.
    StorageID id;
    // the device id of the storage.
    int device_id;
    // maximum size of storage requested.
    size_t max_bytes{0};
    // node index that released it last time
    uint32_t released_by_node{0};
  };
  // scale used for rough match
  size_t match_range_;
  // whether use color based match algorithm
  uint32_t num_match_color_{1};
  // the size of each dtype
  std::vector<size_t> dtype_size_dict_;
  // free list of storage entry
  std::multimap<size_t, StorageEntry*> free_;
  // all the storage resources available
  std::vector<std::unique_ptr<StorageEntry> > data_;
  // color of nodes in the graph, used for auxiliary policy making.
  std::vector<uint32_t> node_color_;
  // internal indexed graph
  const IndexedGraph* idx_;
};

/*
 * Internal method to perform the memory allocation for a graph
 * */
size_t AllocMemory(const Graph& ret, const IndexedGraph& idx,
                   const std::pair<uint32_t, uint32_t>& node_range,
                   StorageVector* storage_ptr,
                   std::vector<int>* storage_inplace_index_ptr,
                   const std::vector<uint32_t>& entry_ref_count,
                   GraphAllocator* allocator) {
  static auto& finplace_option = Op::GetAttr<FInplaceOption>("FInplaceOption");
  static auto& finplace_identity = Op::GetAttr<FInplaceIdentity>("FInplaceIdentity");

  // Get reference
  auto &storage = *storage_ptr;
  auto &storage_inplace_index = *storage_inplace_index_ptr;

  // Get attributes from the graph
  const ShapeVector& shape_vec = ret.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = ret.GetAttr<DTypeVector>("dtype");
  const DeviceVector* device_vec = nullptr;

  if (ret.attrs.count("device") != 0) {
    device_vec = &(ret.GetAttr<DeviceVector>("device"));
  }
  size_t num_not_allocated = 0;
  std::vector<GraphAllocator::StorageID> storage_ref_count(idx.num_node_entries(), 0);

  for (uint32_t nid = node_range.first; nid < node_range.second; ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    // check inplace option
    if (finplace_option.count(inode.source->op()) != 0) {
      auto inplace_pairs = finplace_option[inode.source->op()](inode.source->attrs);
      std::vector<bool> identity;
      if (finplace_identity.count(inode.source->op()) != 0) {
        identity = finplace_identity[inode.source->op()](inode.source->attrs);
        CHECK_EQ(identity.size(), inplace_pairs.size())
            << "FInplaceOption and FInplaceIdentity returned vectors of different "
            << "size for operator " << inode.source->op()->name;
      } else {
        identity = std::vector<bool>(inplace_pairs.size(), false);
      }
      std::vector<bool> taken(inode.inputs.size(), false);
      for (size_t ipair = 0; ipair < inplace_pairs.size(); ++ipair) {
        const auto& kv = inplace_pairs[ipair];
        uint32_t eid_out = idx.entry_id(nid, kv.second);
        uint32_t eid_in = idx.entry_id(inode.inputs[kv.first]);
        auto sid_out = storage[eid_out];
        auto sid_in = storage[eid_in];
        if (taken[kv.first] == false &&
            sid_out == GraphAllocator::kBadStorageID &&
            sid_in >= 0 &&
            (storage_ref_count[sid_in] == 1 || identity[ipair]) &&
            entry_ref_count[eid_out] > 0 &&
            shape_vec[eid_out].Size() == shape_vec[eid_in].Size() &&
            dtype_vec[eid_out] == dtype_vec[eid_in]) {
          // inplace optimization
          taken[kv.first] = true;
          storage[eid_out] = sid_in;
          // Reuse storage for output and add ref count of output
          // to storage. This will get substracted later in free
          // input section.
          storage_ref_count[sid_in] += entry_ref_count[eid_out];
          storage_inplace_index[eid_out] = kv.first;
        } else {
          if (dmlc::GetEnv("USE_COALESCE_SOFTMAX", false)) {
            if (!logged_use_coalesce_softmax) {
              LOG(INFO) << "MXNet will be coalescing the forward and backward "
                        << "pass of the softmax layer. "
                        << "This implies that the use of `forward_backward` is forbidden.";
              logged_use_coalesce_softmax = true;
            }
            if (storage_ref_count[sid_in] == 2 && 
                inode.source->attrs.op->name == "_backward_SoftmaxOutput") {
              taken[kv.first] = true;
              storage[eid_out] = sid_in;
              storage_ref_count[sid_in] += entry_ref_count[eid_out];
              storage_inplace_index[eid_out] = kv.first;
            }
          }
        }
      }
    }
    // normal allocation
    const int dev_id = (device_vec != nullptr) ? device_vec->at(nid) : 0;
    // sort output nodes based on size before allocating output
    std::multimap<size_t, uint32_t> eids;
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      // only request memory for kBadStorageID
      if (storage[eid] == GraphAllocator::kBadStorageID) {
        auto &eshape = shape_vec[eid];
        size_t esize = 0;
        if (eshape.ndim() != 0) esize = eshape.Size();
        eids.insert(std::make_pair(esize, eid));
      }
    }
    for (auto rit = eids.rbegin(); rit != eids.rend(); ++rit) {
        uint32_t eid = rit->second;
        auto sid = allocator->Request(dev_id, dtype_vec[eid], shape_vec[eid], nid);
        storage_ref_count[sid] = entry_ref_count[eid];
        storage[eid] = sid;
    }

    // check if certain inputs is ignored.
    static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
    std::vector<uint32_t> ignore_inputs;
    if (fignore_inputs.count(inode.source->op()) != 0) {
      ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
      std::sort(ignore_inputs.begin(), ignore_inputs.end());
    }
    // then free inputs
    for (size_t i = 0; i < inode.inputs.size(); ++i) {
      // ref counter of ignored input is already decreased.
      if (std::binary_search(ignore_inputs.begin(), ignore_inputs.end(), i)) continue;
      const auto& e = inode.inputs[i];
      uint32_t eid = idx.entry_id(e);
      auto sid = storage[eid];
      // storage_ref_count == 0 means it is taken by inplace op
      if (sid < 0) continue;
      // if we decrease it to zero, means we are ready to relase
      --storage_ref_count[sid];
      if (storage_ref_count[sid] == 0) {
        allocator->Release(sid, nid);
      }

      // if (idx[e.node_id].source->attrs.name == "decoder_rnn_l0_t6_i2h") {
      //   std::cout << "Node decoder_rnn_l0_t6_i2h has a " << std::endl
      //             << "\t""storage_ref_count of " 
      //             << storage_ref_count[eid] << std::endl;
      // }
    }
    // check if there are outputs that can be freeded immediately
    // these output are not referenced by any operator.
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      auto sid = storage[eid];
      if (sid >= 0 && storage_ref_count[sid] == 0) {
        allocator->Release(sid, nid);
        // use -2 to indicate that the node was never touched.
        storage_inplace_index[eid] = -2;
      }
      if (storage[eid] == GraphAllocator::kBadStorageID) {
        ++num_not_allocated;
      }

    }
  }
  return num_not_allocated;
}


// function to plan memory
Graph PlanMemory(Graph ret) {
  // setup ref counter
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
  std::pair<uint32_t, uint32_t> node_range = {0, idx.num_nodes()};
  if (ret.attrs.count("node_range")) {
    node_range = ret.MoveCopyAttr<std::pair<uint32_t, uint32_t> >("node_range");
  }
  // reference counter of each node
  std::vector<uint32_t> ref_count;
  // step 1: initialize reference count
  if (ret.attrs.count("ref_count") != 0) {
    ref_count = ret.MoveCopyAttr<std::vector<uint32_t> >("ref_count");
  } else {
    ref_count.resize(idx.num_node_entries(), 0);
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      const auto& inode = idx[nid];
      if (inode.source->is_variable()) continue;
      for (const auto& e : inode.inputs) {
        ++ref_count[idx.entry_id(e)];

        // if (idx[e.node_id].source->attrs.name == 
        //     "encoder_birnn_reverse_l0_t10_nonlin_block") {
        //   LOG(INFO) << "Encoder Non-Lin Block Oedge " << e.index << " "
        //             << "is referenced by "
        //             << inode.source->attrs.name;
        // }
        // if (idx[e.node_id].source->attrs.name == 
        //     "decoder_rnn_l0_t8_i2h") {
        //   LOG(INFO) << "I2H Oedge " << e.index << " "
        //             << "is referenced by "
        //             << inode.source->attrs.name;
        // }
        // if (idx[e.node_id].source->attrs.name == 
        //     "decoder_rnn_l0_t8_nonlin_block") {
        //   LOG(INFO) << "Decoder Non-Lin Block Oedge " << e.index << " "
        //             << "is referenced by "
        //             << inode.source->attrs.name;
        // }
      }
      // no dataflow dependency is needed for those are ignored.
      // revoke the dependency counter.
      if (fignore_inputs.count(inode.source->op()) != 0) {
        auto ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
        for (uint32_t i : ignore_inputs) {
          --ref_count[idx.entry_id(inode.inputs[i])];
        }
      }
    }
    for (const auto& e : idx.outputs()) {
      ++ref_count[idx.entry_id(e)];
    }
  }
  // step 2: allocate memory.
  StorageVector storage;
  if (ret.attrs.count("storage") != 0) {
    storage = ret.MoveCopyAttr<StorageVector>("storage");
  } else {
    storage.resize(idx.num_node_entries(), -1);
  }

  // Search the best NNVM_EXEC_MATCH_RANGE parameter. This is turned off by default
  size_t min_allocated_bytes = -1;
  size_t max_match_range = dmlc::GetEnv("NNVM_EXEC_MATCH_RANGE", 2);
  size_t min_match_range =
         dmlc::GetEnv("NNVM_AUTO_SEARCH_MATCH_RANGE", false) ? 1 : max_match_range;
  for (size_t match_range = min_match_range; match_range <= max_match_range; match_range *= 2) {
    // Make a copy of related fields
    StorageVector storage_vec(storage);
    std::vector<int> storage_inplace_index(idx.num_node_entries(), -1);

    // the allocator
    GraphAllocator allocator(&idx, match_range);

    // number of entries that are not statically allocated.
    size_t storage_num_not_allocated =
      AllocMemory(ret, idx, node_range, &storage_vec, &storage_inplace_index,
                  ref_count, &allocator);
    size_t storage_allocated_bytes = allocator.TotalAllocBytes();

    // Choose the plan which leads to minimal memory usage
    if (min_allocated_bytes > storage_allocated_bytes) {
      ret.attrs["storage_id"] = std::make_shared<any>(std::move(storage_vec));
      ret.attrs["storage_inplace_index"] = std::make_shared<any>(std::move(storage_inplace_index));
      ret.attrs["storage_allocated_bytes"] = std::make_shared<any>(storage_allocated_bytes);
      ret.attrs["storage_num_not_allocated"] = std::make_shared<any>(storage_num_not_allocated);
      min_allocated_bytes = storage_allocated_bytes;
    }

    if (max_match_range == 0) {
      break;
    }
  }
  return ret;
}

NNVM_REGISTER_PASS(PlanMemory)
.describe("Plan the memory allocation of each node entries.")
.set_body(PlanMemory)
.set_change_graph(false)
.depend_graph_attr("dtype")
.depend_graph_attr("shape")
.provide_graph_attr("storage_id")
.provide_graph_attr("storage_inplace_index");

}  // namespace
}  // namespace pass
}  // namespace nnvm
