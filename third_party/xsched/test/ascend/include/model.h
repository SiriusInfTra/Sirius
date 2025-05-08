#pragma once

#include <list>
#include <string>
#include <memory>
#include <unordered_map>

#include <acl/acl.h>

struct AclSlice
{
    uint32_t model_id;
    aclmdlDesc *model_desc;

    size_t input_size;
    size_t output_size;
    aclmdlDataset *input_dataset;
    aclmdlDataset *output_dataset;
    aclDataBuffer *input_data_buffer;
    aclDataBuffer *output_data_buffer;

    void *input_dev_ptr;
    void *output_dev_ptr;
    void *input_host_ptr;
    void *output_host_ptr;
};

class AclModel
{
public:
    AclModel(const std::string &model_dir);
    ~AclModel();

    void Enqueue(aclrtStream stream);
    void Execute(aclrtStream stream);
    static void InitResource();

private:
    static const int32_t kDeviceId = 0;
    static aclrtContext context_;
    std::list<AclSlice *> slices_;

    void LoadSlice(const std::string &slice_path);
};
