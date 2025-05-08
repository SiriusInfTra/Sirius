#include <fstream>

#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "utils/xassert.h"

aclrtContext AclModel::context_;

AclModel::AclModel(const std::string &model_dir)
{
    InitResource();
    while (true) {
        std::string slice_path = model_dir + "/" +
            std::to_string(slices_.size()) + ".om";
        std::ifstream file(slice_path, std::ios::binary);
        if (!file.good()) break;

        XINFO("loading slice %3ld", slices_.size());
        LoadSlice(slice_path);
    }
    XINFO("load done");
}

AclModel::~AclModel()
{
    if (slices_.size() == 0) return;
    // free input
    free(slices_.front()->input_host_ptr);
    ACL_ASSERT(aclrtFree(slices_.front()->input_dev_ptr));
    ACL_ASSERT(aclmdlDestroyDataset(slices_.front()->input_dataset));

    while (slices_.size() > 0) {
        AclSlice *slice = slices_.front();
        slices_.pop_front();
        // free output
        free(slice->output_host_ptr);
        ACL_ASSERT(aclrtFree(slice->output_dev_ptr));
        ACL_ASSERT(aclmdlDestroyDataset(slice->output_dataset));
        // free model
        ACL_ASSERT(aclmdlDestroyDesc(slice->model_desc));
        ACL_ASSERT(aclmdlUnload(slice->model_id));
        delete slice;
    }
}

void AclModel::Execute(aclrtStream stream)
{
    Enqueue(stream);
    ACL_ASSERT(aclrtSynchronizeStream(stream));
}

void AclModel::Enqueue(aclrtStream stream)
{
    ACL_ASSERT(aclrtSetCurrentContext(context_));
    // ACL_ASSERT(aclrtMemcpyAsync(slices_.front()->input_dev_ptr,
    //                             slices_.front()->input_size,
    //                             slices_.front()->input_host_ptr,
    //                             slices_.front()->input_size,
    //                             ACL_MEMCPY_HOST_TO_DEVICE, stream));
    for (auto slice : slices_) {
        ACL_ASSERT(aclmdlExecuteAsync(slice->model_id,
            slice->input_dataset, slice->output_dataset, stream));
    }
    // ACL_ASSERT(aclrtMemcpyAsync(slices_.back()->output_host_ptr,
    //                             slices_.back()->output_size,
    //                             slices_.back()->output_dev_ptr,
    //                             slices_.back()->output_size,
    //                             ACL_MEMCPY_DEVICE_TO_HOST, stream));
}

void AclModel::InitResource()
{
    static bool initialized = false;
    if (initialized) return;
    initialized = true;
    ACL_ASSERT(aclInit(""));
    ACL_ASSERT(aclrtSetDevice(kDeviceId));
    ACL_ASSERT(aclrtCreateContext(&context_, kDeviceId));
}

void AclModel::LoadSlice(const std::string &slice_path)
{
    uint32_t model_id;
    ACL_ASSERT(aclmdlLoadFromFile(slice_path.c_str(), &model_id));
    aclmdlDesc *model_desc = aclmdlCreateDesc();
    ACL_ASSERT(aclmdlGetDesc(model_desc, model_id));

    aclmdlDataset *input_dataset = aclmdlCreateDataset();
    size_t input_size = aclmdlGetInputSizeByIndex(model_desc, 0);
    void *input_dev_ptr = nullptr;
    void *input_host_ptr = nullptr;
    aclDataBuffer *input_data_buf = nullptr;

    if (slices_.size() == 0) {
        ACL_ASSERT(aclrtMalloc(&input_dev_ptr, input_size,
                               ACL_MEM_MALLOC_HUGE_FIRST));
        input_host_ptr = malloc(input_size);
        input_data_buf = aclCreateDataBuffer(input_dev_ptr, input_size);
    } else {
        XASSERT(input_size == slices_.back()->output_size,
                "input size not match the output size of last slice");
        input_dev_ptr = slices_.back()->output_dev_ptr;
        input_host_ptr = slices_.back()->output_host_ptr;
        input_data_buf = slices_.back()->output_data_buffer;
    }
    
    ACL_ASSERT(aclmdlAddDatasetBuffer(input_dataset, input_data_buf));

    aclmdlDataset *output_dataset = aclmdlCreateDataset();
    size_t output_size = aclmdlGetOutputSizeByIndex(model_desc, 0);
    void *output_dev_ptr = nullptr;
    void *output_host_ptr = malloc(output_size);
    ACL_ASSERT(aclrtMalloc(&output_dev_ptr, output_size,
                           ACL_MEM_MALLOC_HUGE_FIRST));
    aclDataBuffer *output_data_buf = aclCreateDataBuffer(output_dev_ptr,
                                                         output_size);
    ACL_ASSERT(aclmdlAddDatasetBuffer(output_dataset, output_data_buf));

    AclSlice *slice = new AclSlice {
        .model_id = model_id,
        .model_desc = model_desc,
        .input_size = input_size,
        .output_size = output_size,
        .input_dataset = input_dataset,
        .output_dataset = output_dataset,
        .input_data_buffer = input_data_buf,
        .output_data_buffer = output_data_buf,
        .input_dev_ptr = input_dev_ptr,
        .output_dev_ptr = output_dev_ptr,
        .input_host_ptr = input_host_ptr,
        .output_host_ptr = output_host_ptr,
    };

    slices_.push_back(slice);
}
