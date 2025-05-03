#include <server/logging_as_glog.h>
#include <server/model_store/infer_model_store.h>
#include <server/tvm/graph.h>
#include <server/tvm/executor.h>

#include <common/tensor/shape_helper.h>

#include <tvm/runtime/ndarray.h>

#include <boost/range/irange.hpp>
#include <boost/format.hpp>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdlib>


namespace colserve {
namespace tvm {

std::string TVMGraph::mod_json = "mod.json";
std::string TVMGraph::mod_so = "mod.so";
std::string TVMGraph::mod_params = "mod.params";
std::string TVMGraph::mod_group = "mod.group";

TVMGraph::TVMGraph(
    size_t rank, Model *model,
    const std::string &model_name,
    const std::filesystem::path &model_path,
    const std::string &graph_json,
    const std::string &group_txt,
    const ::tvm::runtime::Module mod,
    const std::string &params_file) 
    : model_rank_(rank), infer_model_(model), 
      model_name_(model_name), model_path_(model_path),
      module_(mod) {
  LoadGraph(graph_json);
  LoadParams(params_file);
  SetupStorage();
}

TVMGraph::TVMGraph(
    size_t rank, Model *model,
    const std::string &model_name,
    const std::filesystem::path &model_path,
    const std::string &graph_json,
    const std::string &group_txt,
    const ::tvm::runtime::Module mod,
    const std::map<std::string, TVMArray> &params) 
    : model_rank_(rank), infer_model_(model), 
      model_name_(model_name), model_path_(model_path),
      module_(mod) {
  LoadGraph(graph_json);
  LoadParams(params);
  SetupStorage();
}

void TVMGraph::LoadGraph(const std::string &graph_json) {
  std::ifstream graph_json_ifs{graph_json};
  std::string graph_json_str{
      (std::istreambuf_iterator<char>(graph_json_ifs)),
      std::istreambuf_iterator<char>()
  };
  std::istringstream is(graph_json_str);
  dmlc::JSONReader reader(&is);
  this->Load(&reader);
  for (size_t i = 0; i < nodes_.size(); i++) {
    node_map_[nodes_[i].name] = i;
  }
  for (size_t i = 0; i < input_nodes_.size(); i++) {
    auto nid = input_nodes_[i];
    std::string &name = nodes_[nid].name;
    input_map_[name] = i;
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    auto nid = outputs_[i].node_id;
    std::string &name = nodes_[nid].name;
    output_map_[name] = i;
  }
}

void TVMGraph::LoadParams(const std::string &params_file) {
  auto tvm_params = LoadParamsAsTVMArray(params_file);
  LoadParams(tvm_params);
}

void TVMGraph::LoadParams(const std::map<std::string, TVMArray> &params) {
  for (const auto &p : params) {
    CHECK(input_map_.count(p.first)) << "cannot find " << p.first   
                                     <<  " in model parameter";
    auto it = node_map_.find(p.first);
    auto node_eid = entry_id(it->second, 0);
    host_params_[node_eid] = p.second.CopyTo(::tvm::Device{kDLCUDAHost, 0});
  }
}

void TVMGraph::SetupStorage() {
  std::vector<DLDataType> vtype;
  for (const std::string &s_type : attrs_.dltype) {
    vtype.push_back(::tvm::runtime::String2DLDataType(s_type));
  }

  // ignore the storage scope, as we do not use it
  for (auto i : boost::irange(attrs_.shape.size())) {
    uint32_t sid = attrs_.storage_id[i];
    if (sid >= storage_pool_entries_.size()) {
      storage_pool_entries_.resize(sid + 1, {0});
    } else {
      CHECK_EQ(storage_pool_entries_[sid].is_param_entry, false)
          << "parameter storage " << sid << " cannot be reused";
    }
    CHECK(CheckNullLinkedParam(module_, sid));
    CHECK(attrs_.storage_scope.empty()
          || attrs_.storage_scope[sid].find("texture") == std::string::npos)
        << "unsuported texture memory";

    DLDataType t = vtype[i];
    auto bytes = sta::ComputeStorageNbytes(attrs_.shape[i], t);
    storage_pool_entries_[sid].nbytes = 
        std::max(storage_pool_entries_[sid].nbytes, bytes);
  }

  param_storage_nbytes_ = 0;
  buffer_storage_nbytes_ = 0;
  for (auto sid : boost::irange(storage_pool_entries_.size())) {
    const auto &pit = storage_pool_entries_[sid];
    if (pit.is_param_entry) {
      param_storage_nbytes_ += pit.nbytes;
    } else {
      buffer_storage_nbytes_ += pit.nbytes;
    }
  }

  // SetupHostPinnedIOStorage();

  if (Config::use_shared_tensor_infer
      && Config::better_alloc) {
    SetupStorageGroup();
  }

  auto model_name_without_dup_id = 
      GetModelNameWithoutDuplicatedId(model_name_);

  static std::set<std::string> logged;
  if (!logged.count((model_name_without_dup_id))) {
    logged.insert(model_name_without_dup_id);
    LOG(INFO) << "[Executor] " << model_name_without_dup_id
              << " params " << sta::PrintByte(param_storage_nbytes_)
              << " intermediate " << sta::PrintByte(buffer_storage_nbytes_);
  }
}

void TVMGraph::SetupStorageGroup() {
  CHECK(Config::use_shared_tensor_infer && Config::better_alloc);

  std::vector<uint32_t> storage_alloc_order;
  if (Config::group_param_load) {
    std::vector<bool> storage_record(storage_pool_entries_.size(), false);
    for (auto & p : host_params_) {
      auto sid = attrs_.storage_id[p.first];
      storage_alloc_order.push_back(sid);
      storage_record[sid] = true;
    }
    for (auto i : boost::irange(storage_pool_entries_.size())) {
      if (!storage_record[i]) {
        storage_alloc_order.push_back(i);
      }
    }
  } else {
    for (auto i : boost::irange(storage_pool_entries_.size())) {
      storage_alloc_order.push_back(i);
    }
  }
  storage_alloc_order_ = storage_alloc_order;

  // TO FIX: 
  //    garrentee consistent with the storage order
  //    this is affected by group_param_load
  //    currently, we assume group_param_load is always true
  SetupParamGroupPartition((model_path_ / TVMGraph::mod_group).string());
}

void TVMGraph::SetupParamGroupPartition(const std::string &path) {
  auto group_file_path = model_path_ / TVMGraph::mod_group;

  if (Config::group_param_dump) {
    DumpParamGroupPartition();
  }

  std::ifstream ifs(group_file_path);
  CHECK(ifs.is_open()) << "Cannot open file " << group_file_path
                       << ", enable `group_param_dump` to generate it";
  LOG_IF(INFO, Config::log_infer_model_init) 
      << "load from mod.group from " << path; 

  std::string buf;
  while (ifs >> buf) {
    storage_group_partition_.push_back(std::stoul(buf));
  }

  model_nbytes_with_group_fragment_ = 0;
  memory_byte_t model_nbytes = 0, fragment_nbytes = 0;
  CHECK_GE(storage_group_partition_.size(), 2);
  CHECK_EQ(storage_group_partition_.front(), 0);
  CHECK_EQ(storage_group_partition_.back(), 
           storage_pool_entries_.size());
  for (auto k : boost::irange(storage_group_partition_.size() - 1)) {
    size_t i = storage_group_partition_[k], 
           j = storage_group_partition_[k + 1],
           group_nbytes = 0;
    std::vector<size_t> offsets;
    for (auto iter = storage_alloc_order_.cbegin() + i; 
        iter != storage_alloc_order_.cbegin() + j; ++iter) {
      auto pool_entry = storage_pool_entries_[*iter];
      auto aligned_nbytes = GetLineAlignedNbytes(pool_entry.nbytes);
      offsets.push_back(group_nbytes);
      group_nbytes += aligned_nbytes;
    }
    storage_group_offsets_.push_back(std::move(offsets));
    storage_group_nbytes_.push_back(group_nbytes);
    model_nbytes += group_nbytes;
    fragment_nbytes += GetMemBlockAlignedNBytes(group_nbytes) - group_nbytes;
    model_nbytes_with_group_fragment_ += GetMemBlockAlignedNBytes(group_nbytes);
  }

  if (Config::group_param_load) {
    for (auto sit = host_params_.begin(); sit != host_params_.end(); ) {
      size_t total_nbytes = 0, off = 0;
      std::vector<uint32_t> param_eids;
      auto eit = sit;
      for (; eit != host_params_.end() 
             && total_nbytes < Config::group_param_load_threshold; eit++) {
        auto &p = *eit;
        auto aligned_nbytes = GetLineAlignedNbytes(
            ::tvm::runtime::GetDataSize(*p.second.operator->()));
        total_nbytes += aligned_nbytes;
        param_eids.push_back(p.first);
        // param_ready_event_ids_[p.first] = host_param_storage_group_.size();
      }
      auto param_group = sta::HostEmpty({static_cast<int64_t>(total_nbytes)}, 
                                        DLDataType{kDLInt, 8, 1}, 
                                        sta::MemType::kInfer);
      for (; sit != eit; sit++) {
        auto &p = *sit;
        std::memcpy(static_cast<char*>(param_group->data) + off, p.second->data,
            ::tvm::runtime::GetDataSize(*p.second.operator->()));
        auto aligned_nbytes = GetLineAlignedNbytes(
            ::tvm::runtime::GetDataSize(*p.second.operator->()));
        off += aligned_nbytes;
      }
      this->host_param_group_.push_back(
          std::make_pair(param_group, param_eids));
    }
  }

  static std::set<std::string> logged;
  auto model_name_without_dup_id = GetModelNameWithoutDuplicatedId(model_name_);
  if (!logged.count((model_name_without_dup_id))) {
    logged.insert(model_name_without_dup_id);
    LOG_IF(INFO, Config::log_infer_model_init) 
        << "[Executor] " << model_name_ << " internal fragment: " 
        << sta::PrintByte(fragment_nbytes) << " / " << sta::PrintByte(model_nbytes)
        << " | model with group fragment "
        << sta::PrintByte(model_nbytes_with_group_fragment_);
  }
}

void TVMGraph::DumpParamGroupPartition() {
  auto group_file_path = model_path_ / TVMGraph::mod_group;

  auto model_nbytes_acc = std::accumulate(
        storage_pool_entries_.begin(), 
        storage_pool_entries_.end(), 0,
        [](int acc, const PoolEntry &entry) {
          return acc + entry.nbytes;
        });
  std::string file_name = (model_path_ / 
      ("nbytes_" + std::to_string(model_nbytes_acc) + ".txt")
    ).string();
  LOG(INFO) << "[Executor] " << model_name_ 
            << " dump storage nbytes to " << file_name;
  if (!std::filesystem::exists(file_name)) {
    std::ofstream ofs(file_name);
    for (auto eid : storage_alloc_order_) {
      ofs << storage_pool_entries_[eid].nbytes << std::endl;
    }
    // for (auto entry : storage_pool_entries_) {
    //   ofs << entry.nbytes << std::endl;
    // }
    ofs.close();
    CHECK_EQ(system(
      ("python util/prepare_model_store/offline_group.py"
       " --storage-nbytes-file "
        + file_name + " --output "
        + group_file_path.string()
      ).c_str()
    ), 0) << " generate storage group failed";
  } else {
    LOG(WARNING) << "[Executor] " << model_name_ 
                  << " storage nbytes file " << file_name 
                  << " already exists, skip dump";
  }
}

std::map<std::string, TVMArray> 
TVMGraph::LoadParamsAsTVMArray(const std::string &params_file) {
  struct stat file_stat;
  int params_fd = open(params_file.c_str(), O_RDONLY);
  fstat(params_fd, &file_stat);
  auto params_ptr = static_cast<const char*>(
      mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, params_fd, 0U));
  std::string params_blob;
  params_blob.reserve(file_stat.st_size);
  params_blob.assign(params_ptr, params_ptr + file_stat.st_size);
  munmap((void*)params_ptr, file_stat.st_size);

  dmlc::MemoryStringStream ms_strm(const_cast<std::string*>(&params_blob));
  dmlc::Stream* strm = &ms_strm;
  uint64_t header, reserved;
  constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;
  CHECK(strm->Read(&header)) << "Invalid parameters file format";
  CHECK(header == kTVMNDArrayListMagic) << "Invalid parameters file format";
  CHECK(strm->Read(&reserved)) << "Invalid parameters file format";

  std::vector<std::string> names;
  CHECK(strm->Read(&names)) << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  CHECK(size == names.size()) << "Invalid parameters file format";
  size_t params_size = 0;
  std::map<std::string, TVMArray> params_ret;
  for (size_t i = 0; i < size; ++i) {
    // The data_entry is allocated on device, 
    // NDArray.load always load the array into CPU.
    TVMArray temp;
    temp.Load(strm);

    params_ret[names[i]] = temp;
    params_size += ::tvm::runtime::GetDataSize(*temp.operator->());
  }
  VLOG(1) << params_file << " " << sta::PrintByte(params_size);
  return params_ret;
}

std::unique_ptr<Executor> TVMGraph::CreateGraphExecutor(
    size_t worker_id, 
    const std::vector<DLDevice> &devs) {
  return std::make_unique<Executor>(*this, worker_id, devs);
}

std::tuple<TVMGraph::ShapeInfo, TVMGraph::DtypeInfo>
TVMGraph::GetInputInfo() const {
  ShapeInfo shape_info;
  DtypeInfo dtype_info;
  for (auto nid : input_nodes_) {
    auto eid = entry_id(nid, 0);
    if (!host_params_.count(eid)) {
      shape_info[nodes_[nid].name] = attrs_.shape[eid];
      dtype_info[nodes_[nid].name] = attrs_.dltype[eid];
    }
  }
  return {shape_info, dtype_info};
}

std::tuple<TVMGraph::ShapeInfo, TVMGraph::DtypeInfo>
    TVMGraph::GetOutputInfo() const {
  ShapeInfo shape_info;
  DtypeInfo dtype_info;
  for (auto e : outputs_) {
    auto nid = e.node_id;
    auto eid = entry_id(e);
    DLOG(INFO) << nodes_[nid].name << " " << nid << " " << eid << ", " 
               << attrs_.shape[eid] << " " << attrs_.dltype[eid];
    shape_info[nodes_[nid].name] = attrs_.shape[eid];
    dtype_info[nodes_[nid].name] = attrs_.dltype[eid];
  }
  return {shape_info, dtype_info};
}

}
}