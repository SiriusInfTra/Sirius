#pragma once

/**
 * 
 * @file ge_error_codes.h
 * 
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INC_EXTERNAL_GE_GE_ERROR_CODES_H_
#define INC_EXTERNAL_GE_GE_ERROR_CODES_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
static const uint32_t ACL_ERROR_GE_PARAM_INVALID = 145000U;
static const uint32_t ACL_ERROR_GE_EXEC_NOT_INIT = 145001U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID = 145002U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_ID_INVALID = 145003U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID = 145006U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID = 145007U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID = 145008U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED = 145009U;
static const uint32_t ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID = 145011U;
static const uint32_t ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID = 145012U;
static const uint32_t ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID = 145013U;
static const uint32_t ACL_ERROR_GE_AIPP_BATCH_EMPTY = 145014U;
static const uint32_t ACL_ERROR_GE_AIPP_NOT_EXIST = 145015U;
static const uint32_t ACL_ERROR_GE_AIPP_MODE_INVALID = 145016U;
static const uint32_t ACL_ERROR_GE_OP_TASK_TYPE_INVALID = 145017U;
static const uint32_t ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID = 145018U;
static const uint32_t ACL_ERROR_GE_PLGMGR_PATH_INVALID = 145019U;
static const uint32_t ACL_ERROR_GE_FORMAT_INVALID = 145020U;
static const uint32_t ACL_ERROR_GE_SHAPE_INVALID = 145021U;
static const uint32_t ACL_ERROR_GE_DATATYPE_INVALID = 145022U;
static const uint32_t ACL_ERROR_GE_MEMORY_ALLOCATION = 245000U;
static const uint32_t ACL_ERROR_GE_MEMORY_OPERATE_FAILED = 245001U;
static const uint32_t ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED = 245002U;
static const uint32_t ACL_ERROR_GE_INTERNAL_ERROR = 545000U;
static const uint32_t ACL_ERROR_GE_LOAD_MODEL = 545001U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED = 545002U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED = 545003U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED = 545004U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED = 545005U;
static const uint32_t ACL_ERROR_GE_EXEC_RELEASE_MODEL_DATA = 545006U;
static const uint32_t ACL_ERROR_GE_COMMAND_HANDLE = 545007U;
static const uint32_t ACL_ERROR_GE_GET_TENSOR_INFO = 545008U;
static const uint32_t ACL_ERROR_GE_UNLOAD_MODEL = 545009U;

#ifdef __cplusplus
}  // namespace ge
#endif
#endif  // INC_EXTERNAL_GE_GE_ERROR_CODES_H_


/**
* @file rt_error_codes.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef __INC_EXTERNEL_RT_ERROR_CODES_H__
#define __INC_EXTERNEL_RT_ERROR_CODES_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define  ACL_RT_SUCCESS    0                               // success
#define  ACL_ERROR_RT_PARAM_INVALID              107000 // param invalid
#define  ACL_ERROR_RT_INVALID_DEVICEID           107001 // invalid device id
#define  ACL_ERROR_RT_CONTEXT_NULL               107002 // current context null
#define  ACL_ERROR_RT_STREAM_CONTEXT             107003 // stream not in current context
#define  ACL_ERROR_RT_MODEL_CONTEXT              107004 // model not in current context
#define  ACL_ERROR_RT_STREAM_MODEL               107005 // stream not in model
#define  ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID    107006 // event timestamp invalid
#define  ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL   107007 // event timestamp reversal
#define  ACL_ERROR_RT_ADDR_UNALIGNED             107008 // memory address unaligned
#define  ACL_ERROR_RT_FILE_OPEN                  107009 // open file failed
#define  ACL_ERROR_RT_FILE_WRITE                 107010 // write file failed
#define  ACL_ERROR_RT_STREAM_SUBSCRIBE           107011 // error subscribe stream
#define  ACL_ERROR_RT_THREAD_SUBSCRIBE           107012 // error subscribe thread
#define  ACL_ERROR_RT_GROUP_NOT_SET              107013 // group not set
#define  ACL_ERROR_RT_GROUP_NOT_CREATE           107014 // group not create
#define  ACL_ERROR_RT_STREAM_NO_CB_REG           107015 // callback not register to stream
#define  ACL_ERROR_RT_INVALID_MEMORY_TYPE        107016 // invalid memory type
#define  ACL_ERROR_RT_INVALID_HANDLE             107017 // invalid handle
#define  ACL_ERROR_RT_INVALID_MALLOC_TYPE        107018 // invalid malloc type
#define  ACL_ERROR_RT_WAIT_TIMEOUT               107019 // wait timeout
#define  ACL_ERROR_RT_TASK_TIMEOUT               107020 // task timeout
#define  ACL_ERROR_RT_SYSPARAMOPT_NOT_SET        107021 // not set sysparamopt

#define  ACL_ERROR_RT_FEATURE_NOT_SUPPORT        207000 // feature not support
#define  ACL_ERROR_RT_MEMORY_ALLOCATION          207001 // memory allocation error
#define  ACL_ERROR_RT_MEMORY_FREE                207002 // memory free error
#define  ACL_ERROR_RT_AICORE_OVER_FLOW           207003 // aicore over flow
#define  ACL_ERROR_RT_NO_DEVICE                  207004 // no device
#define  ACL_ERROR_RT_RESOURCE_ALLOC_FAIL        207005 // resource alloc fail
#define  ACL_ERROR_RT_NO_PERMISSION              207006 // no permission
#define  ACL_ERROR_RT_NO_EVENT_RESOURCE          207007 // no event resource
#define  ACL_ERROR_RT_NO_STREAM_RESOURCE         207008 // no stream resource
#define  ACL_ERROR_RT_NO_NOTIFY_RESOURCE         207009 // no notify resource
#define  ACL_ERROR_RT_NO_MODEL_RESOURCE          207010 // no model resource
#define  ACL_ERROR_RT_NO_CDQ_RESOURCE            207011 // no cdq resource
#define  ACL_ERROR_RT_OVER_LIMIT                 207012 // over limit
#define  ACL_ERROR_RT_QUEUE_EMPTY                207013 // queue is empty
#define  ACL_ERROR_RT_QUEUE_FULL                 207014 // queue is full
#define  ACL_ERROR_RT_REPEATED_INIT              207015 // repeated init
#define  ACL_ERROR_RT_AIVEC_OVER_FLOW            207016 // aivec over flow
#define  ACL_ERROR_RT_OVER_FLOW                  207017 // common over flow
#define  ACL_ERROR_RT_DEVIDE_OOM                 207018 // device oom

#define  ACL_ERROR_RT_INTERNAL_ERROR             507000 // runtime internal error
#define  ACL_ERROR_RT_TS_ERROR                   507001 // ts internel error
#define  ACL_ERROR_RT_STREAM_TASK_FULL           507002 // task full in stream
#define  ACL_ERROR_RT_STREAM_TASK_EMPTY          507003 // task empty in stream
#define  ACL_ERROR_RT_STREAM_NOT_COMPLETE        507004 // stream not complete
#define  ACL_ERROR_RT_END_OF_SEQUENCE            507005 // end of sequence
#define  ACL_ERROR_RT_EVENT_NOT_COMPLETE         507006 // event not complete
#define  ACL_ERROR_RT_CONTEXT_RELEASE_ERROR      507007 // context release error
#define  ACL_ERROR_RT_SOC_VERSION                507008 // soc version error
#define  ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT      507009 // task type not support
#define  ACL_ERROR_RT_LOST_HEARTBEAT             507010 // ts lost heartbeat
#define  ACL_ERROR_RT_MODEL_EXECUTE              507011 // model execute failed
#define  ACL_ERROR_RT_REPORT_TIMEOUT             507012 // report timeout
#define  ACL_ERROR_RT_SYS_DMA                    507013 // sys dma error
#define  ACL_ERROR_RT_AICORE_TIMEOUT             507014 // aicore timeout
#define  ACL_ERROR_RT_AICORE_EXCEPTION           507015 // aicore exception
#define  ACL_ERROR_RT_AICORE_TRAP_EXCEPTION      507016 // aicore trap exception
#define  ACL_ERROR_RT_AICPU_TIMEOUT              507017 // aicpu timeout
#define  ACL_ERROR_RT_AICPU_EXCEPTION            507018 // aicpu exception
#define  ACL_ERROR_RT_AICPU_DATADUMP_RSP_ERR     507019 // aicpu datadump response error
#define  ACL_ERROR_RT_AICPU_MODEL_RSP_ERR        507020 // aicpu model operate response error
#define  ACL_ERROR_RT_PROFILING_ERROR            507021 // profiling error
#define  ACL_ERROR_RT_IPC_ERROR                  507022 // ipc error
#define  ACL_ERROR_RT_MODEL_ABORT_NORMAL         507023 // model abort normal
#define  ACL_ERROR_RT_KERNEL_UNREGISTERING       507024 // kernel unregistering
#define  ACL_ERROR_RT_RINGBUFFER_NOT_INIT        507025 // ringbuffer not init
#define  ACL_ERROR_RT_RINGBUFFER_NO_DATA         507026 // ringbuffer no data
#define  ACL_ERROR_RT_KERNEL_LOOKUP              507027 // kernel lookup error
#define  ACL_ERROR_RT_KERNEL_DUPLICATE           507028 // kernel register duplicate
#define  ACL_ERROR_RT_DEBUG_REGISTER_FAIL        507029 // debug register failed
#define  ACL_ERROR_RT_DEBUG_UNREGISTER_FAIL      507030 // debug unregister failed
#define  ACL_ERROR_RT_LABEL_CONTEXT              507031 // label not in current context
#define  ACL_ERROR_RT_PROGRAM_USE_OUT            507032 // program register num use out
#define  ACL_ERROR_RT_DEV_SETUP_ERROR            507033 // device setup error
#define  ACL_ERROR_RT_VECTOR_CORE_TIMEOUT        507034 // vector core timeout
#define  ACL_ERROR_RT_VECTOR_CORE_EXCEPTION      507035 // vector core exception
#define  ACL_ERROR_RT_VECTOR_CORE_TRAP_EXCEPTION 507036 // vector core trap exception
#define  ACL_ERROR_RT_CDQ_BATCH_ABNORMAL         507037 // cdq alloc batch abnormal
#define  ACL_ERROR_RT_DIE_MODE_CHANGE_ERROR      507038 // can not change die mode
#define  ACL_ERROR_RT_DIE_SET_ERROR              507039 // single die mode can not set die
#define  ACL_ERROR_RT_INVALID_DIEID              507040 // invalid die id
#define  ACL_ERROR_RT_DIE_MODE_NOT_SET           507041 // die mode not set
#define  ACL_ERROR_RT_AICORE_TRAP_READ_OVERFLOW       507042 // aic trap read overflow
#define  ACL_ERROR_RT_AICORE_TRAP_WRITE_OVERFLOW      507043 // aic trap write overflow
#define  ACL_ERROR_RT_VECTOR_CORE_TRAP_READ_OVERFLOW  507044 // aiv trap read overflow
#define  ACL_ERROR_RT_VECTOR_CORE_TRAP_WRITE_OVERFLOW 507045 // aiv trap write overflow
#define  ACL_ERROR_RT_STREAM_SYNC_TIMEOUT        507046 // stream sync time out
#define  ACL_ERROR_RT_EVENT_SYNC_TIMEOUT         507047 // event sync time out
#define  ACL_ERROR_RT_FFTS_PLUS_TIMEOUT          507048 // ffts+ timeout
#define  ACL_ERROR_RT_FFTS_PLUS_EXCEPTION        507049 // ffts+ exception
#define  ACL_ERROR_RT_FFTS_PLUS_TRAP_EXCEPTION   507050 // ffts+ trap exception
#define  ACL_ERROR_RT_SEND_MSG                   507051 // hdc send msg fail
#define  ACL_ERROR_RT_COPY_DATA                  507052 // copy data fail
#define  ACL_ERROR_RT_DRV_INTERNAL_ERROR         507899 // drv internal error
#define  ACL_ERROR_RT_AICPU_INTERNAL_ERROR       507900 // aicpu internal error
#define  ACL_ERROR_RT_SOCKET_CLOSE               507901 // hdc disconnect

#ifdef __cplusplus
}
#endif
#endif // __INC_EXTERNEL_RT_ERROR_CODES_H__


/**
* @file acl_base.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_BASE_H_
#define INC_EXTERNAL_ACL_ACL_BASE_H_

#include <stdint.h>
#include <stddef.h>
// #include "error_codes/rt_error_codes.h"
// #include "error_codes/ge_error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY _declspec(dllexport)
#else
#define ACL_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define ACL_FUNC_VISIBILITY
#endif
#endif

#ifdef __GNUC__
#define ACL_DEPRECATED __attribute__((deprecated))
#define ACL_DEPRECATED_MESSAGE(message) __attribute__((deprecated(message)))
#elif defined(_MSC_VER)
#define ACL_DEPRECATED __declspec(deprecated)
#define ACL_DEPRECATED_MESSAGE(message) __declspec(deprecated(message))
#else
#define ACL_DEPRECATED
#define ACL_DEPRECATED_MESSAGE(message)
#endif

typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclrtContext;
typedef int aclError;
typedef uint16_t aclFloat16;
typedef struct aclDataBuffer aclDataBuffer;
typedef struct aclTensorDesc aclTensorDesc;
typedef void *aclrtAllocatorDesc;
typedef void *aclrtAllocator;
typedef void *aclrtAllocatorBlock;
typedef void *aclrtAllocatorAddr;

static const int ACL_ERROR_NONE = 0;
static const int ACL_SUCCESS = 0;

static const int ACL_ERROR_INVALID_PARAM = 100000;
static const int ACL_ERROR_UNINITIALIZE = 100001;
static const int ACL_ERROR_REPEAT_INITIALIZE = 100002;
static const int ACL_ERROR_INVALID_FILE = 100003;
static const int ACL_ERROR_WRITE_FILE = 100004;
static const int ACL_ERROR_INVALID_FILE_SIZE = 100005;
static const int ACL_ERROR_PARSE_FILE = 100006;
static const int ACL_ERROR_FILE_MISSING_ATTR = 100007;
static const int ACL_ERROR_FILE_ATTR_INVALID = 100008;
static const int ACL_ERROR_INVALID_DUMP_CONFIG = 100009;
static const int ACL_ERROR_INVALID_PROFILING_CONFIG = 100010;
static const int ACL_ERROR_INVALID_MODEL_ID = 100011;
static const int ACL_ERROR_DESERIALIZE_MODEL = 100012;
static const int ACL_ERROR_PARSE_MODEL = 100013;
static const int ACL_ERROR_READ_MODEL_FAILURE = 100014;
static const int ACL_ERROR_MODEL_SIZE_INVALID = 100015;
static const int ACL_ERROR_MODEL_MISSING_ATTR = 100016;
static const int ACL_ERROR_MODEL_INPUT_NOT_MATCH = 100017;
static const int ACL_ERROR_MODEL_OUTPUT_NOT_MATCH = 100018;
static const int ACL_ERROR_MODEL_NOT_DYNAMIC = 100019;
static const int ACL_ERROR_OP_TYPE_NOT_MATCH = 100020;
static const int ACL_ERROR_OP_INPUT_NOT_MATCH = 100021;
static const int ACL_ERROR_OP_OUTPUT_NOT_MATCH = 100022;
static const int ACL_ERROR_OP_ATTR_NOT_MATCH = 100023;
static const int ACL_ERROR_OP_NOT_FOUND = 100024;
static const int ACL_ERROR_OP_LOAD_FAILED = 100025;
static const int ACL_ERROR_UNSUPPORTED_DATA_TYPE = 100026;
static const int ACL_ERROR_FORMAT_NOT_MATCH = 100027;
static const int ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED = 100028;
static const int ACL_ERROR_KERNEL_NOT_FOUND = 100029;
static const int ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED = 100030;
static const int ACL_ERROR_KERNEL_ALREADY_REGISTERED = 100031;
static const int ACL_ERROR_INVALID_QUEUE_ID = 100032;
static const int ACL_ERROR_REPEAT_SUBSCRIBE = 100033;
static const int ACL_ERROR_STREAM_NOT_SUBSCRIBE = 100034;
static const int ACL_ERROR_THREAD_NOT_SUBSCRIBE = 100035;
static const int ACL_ERROR_WAIT_CALLBACK_TIMEOUT = 100036;
static const int ACL_ERROR_REPEAT_FINALIZE = 100037;
static const int ACL_ERROR_NOT_STATIC_AIPP = 100038;
static const int ACL_ERROR_COMPILING_STUB_MODE = 100039;
static const int ACL_ERROR_GROUP_NOT_SET = 100040;
static const int ACL_ERROR_GROUP_NOT_CREATE = 100041;
static const int ACL_ERROR_PROF_ALREADY_RUN = 100042;
static const int ACL_ERROR_PROF_NOT_RUN = 100043;
static const int ACL_ERROR_DUMP_ALREADY_RUN = 100044;
static const int ACL_ERROR_DUMP_NOT_RUN = 100045;
static const int ACL_ERROR_PROF_REPEAT_SUBSCRIBE = 148046;
static const int ACL_ERROR_PROF_API_CONFLICT = 148047;
static const int ACL_ERROR_INVALID_MAX_OPQUEUE_NUM_CONFIG = 148048;
static const int ACL_ERROR_INVALID_OPP_PATH = 148049;
static const int ACL_ERROR_OP_UNSUPPORTED_DYNAMIC = 148050;
static const int ACL_ERROR_RELATIVE_RESOURCE_NOT_CLEARED = 148051;
static const int ACL_ERROR_UNSUPPORTED_JPEG = 148052;

static const int ACL_ERROR_BAD_ALLOC = 200000;
static const int ACL_ERROR_API_NOT_SUPPORT = 200001;
static const int ACL_ERROR_INVALID_DEVICE = 200002;
static const int ACL_ERROR_MEMORY_ADDRESS_UNALIGNED = 200003;
static const int ACL_ERROR_RESOURCE_NOT_MATCH = 200004;
static const int ACL_ERROR_INVALID_RESOURCE_HANDLE = 200005;
static const int ACL_ERROR_FEATURE_UNSUPPORTED = 200006;
static const int ACL_ERROR_PROF_MODULES_UNSUPPORTED = 200007;

static const int ACL_ERROR_STORAGE_OVER_LIMIT = 300000;

static const int ACL_ERROR_INTERNAL_ERROR = 500000;
static const int ACL_ERROR_FAILURE = 500001;
static const int ACL_ERROR_GE_FAILURE = 500002;
static const int ACL_ERROR_RT_FAILURE = 500003;
static const int ACL_ERROR_DRV_FAILURE = 500004;
static const int ACL_ERROR_PROFILING_FAILURE = 500005;

#define ACL_TENSOR_SHAPE_RANGE_NUM 2
#define ACL_TENSOR_VALUE_RANGE_NUM 2
#define ACL_UNKNOWN_RANK 0xFFFFFFFFFFFFFFFE

typedef enum {
    ACL_DT_UNDEFINED = -1,
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_INT16 = 6,
    ACL_UINT16 = 7,
    ACL_UINT32 = 8,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
    ACL_STRING = 13,
    ACL_COMPLEX64 = 16,
    ACL_COMPLEX128 = 17,
    ACL_BF16 = 27,
    ACL_UINT1 = 30,
    ACL_COMPLEX32 = 33,
} aclDataType;

typedef enum {
    ACL_FORMAT_UNDEFINED = -1,
    ACL_FORMAT_NCHW = 0,
    ACL_FORMAT_NHWC = 1,
    ACL_FORMAT_ND = 2,
    ACL_FORMAT_NC1HWC0 = 3,
    ACL_FORMAT_FRACTAL_Z = 4,
    ACL_FORMAT_NC1HWC0_C04 = 12,
    ACL_FORMAT_HWCN = 16,
    ACL_FORMAT_NDHWC = 27,
    ACL_FORMAT_FRACTAL_NZ = 29,
    ACL_FORMAT_NCDHW = 30,
    ACL_FORMAT_NDC1HWC0 = 32,
    ACL_FRACTAL_Z_3D = 33,
    ACL_FORMAT_NC = 35,
    ACL_FORMAT_NCL = 47,
} aclFormat;

typedef enum {
    ACL_DEBUG = 0,
    ACL_INFO = 1,
    ACL_WARNING = 2,
    ACL_ERROR = 3,
} aclLogLevel;

typedef enum {
    ACL_MEMTYPE_DEVICE = 0,
    ACL_MEMTYPE_HOST = 1,
    ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT = 2
} aclMemType;

typedef enum {
    ACL_OPT_DETERMINISTIC = 0,
} aclSysParamOpt;

/**
 * @ingroup AscendCL
 * @brief Converts data of type aclFloat16 to data of type float
 *
 * @param value [IN]   Data to be converted
 *
 * @retval Transformed data
 */
ACL_FUNC_VISIBILITY float aclFloat16ToFloat(aclFloat16 value);

/**
 * @ingroup AscendCL
 * @brief Converts data of type float to data of type aclFloat16
 *
 * @param value [IN]   Data to be converted
 *
 * @retval Transformed data
 */
ACL_FUNC_VISIBILITY aclFloat16 aclFloatToFloat16(float value);

/**
 * @ingroup AscendCL
 * @brief create data of aclDataBuffer
 *
 * @param data [IN]    pointer to data
 * @li Need to be managed by the user,
 *  call aclrtMalloc interface to apply for memory,
 *  call aclrtFree interface to release memory
 *
 * @param size [IN]    size of data in bytes
 *
 * @retval pointer to created instance. nullptr if run out of memory
 *
 * @see aclrtMalloc | aclrtFree
 */
ACL_FUNC_VISIBILITY aclDataBuffer *aclCreateDataBuffer(void *data, size_t size);

/**
 * @ingroup AscendCL
 * @brief destroy data of aclDataBuffer
 *
 * @par Function
 *  Only the aclDataBuffer type data is destroyed here.
 *  The memory of the data passed in when the aclDataDataBuffer interface
 *  is called to create aclDataBuffer type data must be released by the user
 *
 * @param  dataBuffer [IN]   pointer to the aclDataBuffer
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclCreateDataBuffer
 */
ACL_FUNC_VISIBILITY aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief update new data of aclDataBuffer
 *
 * @param dataBuffer [OUT]    pointer to aclDataBuffer
 * @li The old data need to be released by the user, otherwise it may occur memory leak leakage
 *  call aclGetDataBufferAddr interface to get old data address
 *  call aclrtFree interface to release memory
 *
 * @param data [IN]    pointer to new data
 * @li Need to be managed by the user,
 *  call aclrtMalloc interface to apply for memory,
 *  call aclrtFree interface to release memory
 *
 * @param size [IN]    size of data in bytes
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMalloc | aclrtFree | aclGetDataBufferAddr
 */
ACL_FUNC_VISIBILITY aclError aclUpdateDataBuffer(aclDataBuffer *dataBuffer, void *data, size_t size);

/**
 * @ingroup AscendCL
 * @brief get data address from aclDataBuffer
 *
 * @param dataBuffer [IN]    pointer to the data of aclDataBuffer
 *
 * @retval data address
 */
ACL_FUNC_VISIBILITY void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief get data size of aclDataBuffer
 *
 * @param  dataBuffer [IN]    pointer to the data of aclDataBuffer
 *
 * @retval data size
 */
ACL_DEPRECATED_MESSAGE("aclGetDataBufferSize is deprecated, use aclGetDataBufferSizeV2 instead")
ACL_FUNC_VISIBILITY uint32_t aclGetDataBufferSize(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief get data size of aclDataBuffer to replace aclGetDataBufferSize
 *
 * @param  dataBuffer [IN]    pointer to the data of aclDataBuffer
 *
 * @retval data size
 */
ACL_FUNC_VISIBILITY size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief get size of aclDataType
 *
 * @param  dataType [IN]    aclDataType data the size to get
 *
 * @retval size of the aclDataType
 */
ACL_FUNC_VISIBILITY size_t aclDataTypeSize(aclDataType dataType);

// interfaces of tensor desc
/**
 * @ingroup AscendCL
 * @brief create data aclTensorDesc
 *
 * @param  dataType [IN]    Data types described by tensor
 * @param  numDims [IN]     the number of dimensions of the shape
 * @param  dims [IN]        the size of the specified dimension
 * @param  format [IN]      tensor format
 *
 * @retval aclTensorDesc pointer.
 * @retval nullptr if param is invalid or run out of memory
 */
ACL_FUNC_VISIBILITY aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
                                                       int numDims,
                                                       const int64_t *dims,
                                                       aclFormat format);

/**
 * @ingroup AscendCL
 * @brief destroy data aclTensorDesc
 *
 * @param desc [IN]     pointer to the data of aclTensorDesc to destroy
 */
ACL_FUNC_VISIBILITY void aclDestroyTensorDesc(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief set tensor shape range for aclTensorDesc
 *
 * @param  desc [OUT]     pointer to the data of aclTensorDesc
 * @param  dimsCount [IN]     the number of dimensions of the shape
 * @param  dimsRange [IN]     the range of dimensions of the shape
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorShapeRange(aclTensorDesc* desc,
                                                    size_t dimsCount,
                                                    int64_t dimsRange[][ACL_TENSOR_SHAPE_RANGE_NUM]);

/**
 * @ingroup AscendCL
 * @brief set value range for aclTensorDesc
 *
 * @param  desc [OUT]     pointer to the data of aclTensorDesc
 * @param  valueCount [IN]     the number of value
 * @param  valueRange [IN]     the range of value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorValueRange(aclTensorDesc* desc,
                                                    size_t valueCount,
                                                    int64_t valueRange[][ACL_TENSOR_VALUE_RANGE_NUM]);
/**
 * @ingroup AscendCL
 * @brief get data type specified by the tensor description
 *
 * @param desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval data type specified by the tensor description.
 * @retval ACL_DT_UNDEFINED if description is null
 */
ACL_FUNC_VISIBILITY aclDataType aclGetTensorDescType(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get data format specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval data format specified by the tensor description.
 * @retval ACL_FORMAT_UNDEFINED if description is null
 */
ACL_FUNC_VISIBILITY aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get tensor size specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval data size specified by the tensor description.
 * @retval 0 if description is null
 */
ACL_FUNC_VISIBILITY size_t aclGetTensorDescSize(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get element count specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval element count specified by the tensor description.
 * @retval 0 if description is null
 */
ACL_FUNC_VISIBILITY size_t aclGetTensorDescElementCount(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get number of dims specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval number of dims specified by the tensor description.
 * @retval 0 if description is null
 * @retval ACL_UNKNOWN_RANK if the tensor dim is -2
 */
ACL_FUNC_VISIBILITY size_t aclGetTensorDescNumDims(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief Get the size of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 *
 * @retval dim specified by the tensor description and index.
 * @retval -1 if description or index is invalid
 */
ACL_DEPRECATED_MESSAGE("aclGetTensorDescDim is deprecated, use aclGetTensorDescDimV2 instead")
ACL_FUNC_VISIBILITY int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index);

/**
 * @ingroup AscendCL
 * @brief Get the size of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 * @param  dimSize [OUT]    size of the specified dim.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize);

/**
 * @ingroup AscendCL
 * @brief Get the range of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 * @param  dimRangeNum [IN]     number of dimRange.
 * @param  dimRange [OUT]       range of the specified dim.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclGetTensorDescDimRange(const aclTensorDesc *desc,
                                                      size_t index,
                                                      size_t dimRangeNum,
                                                      int64_t *dimRange);

/**
 * @ingroup AscendCL
 * @brief set tensor description name
 *
 * @param desc [OUT]       pointer to the instance of aclTensorDesc
 * @param name [IN]        tensor description name
 */
ACL_FUNC_VISIBILITY void aclSetTensorDescName(aclTensorDesc *desc, const char *name);

/**
 * @ingroup AscendCL
 * @brief get tensor description name
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval tensor description name.
 * @retval empty string if description is null
 */
ACL_FUNC_VISIBILITY const char *aclGetTensorDescName(aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief Convert the format in the source aclTensorDesc according to
 * the specified dstFormat to generate a new target aclTensorDesc.
 * The format in the source aclTensorDesc remains unchanged.
 *
 * @param  srcDesc [IN]     pointer to the source tensor desc
 * @param  dstFormat [IN]   destination format
 * @param  dstDesc [OUT]    pointer to the pointer to the destination tensor desc
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclTransTensorDescFormat(const aclTensorDesc *srcDesc, aclFormat dstFormat,
    aclTensorDesc **dstDesc);

/**
 * @ingroup AscendCL
 * @brief Set the storage format specified by the tensor description
 *
 * @param  desc [OUT]     pointer to the instance of aclTensorDesc
 * @param  format [IN]    the storage format
 *
 * @retval ACL_SUCCESS    The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_DEPRECATED_MESSAGE("aclSetTensorStorageFormat is deprecated, use aclSetTensorFormat instead")
ACL_FUNC_VISIBILITY aclError aclSetTensorStorageFormat(aclTensorDesc *desc, aclFormat format);

/**
 * @ingroup AscendCL
 * @brief Set the storage shape specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  numDims [IN]    the number of dimensions of the shape
 * @param  dims [IN]       the size of the specified dimension
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_DEPRECATED_MESSAGE("aclSetTensorStorageShape is deprecated, use aclSetTensorShape instead")
ACL_FUNC_VISIBILITY aclError aclSetTensorStorageShape(aclTensorDesc *desc, int numDims, const int64_t *dims);

/**
 * @ingroup AscendCL
 * @brief Set the format specified by the tensor description
 *
 * @param  desc [OUT]     pointer to the instance of aclTensorDesc
 * @param  format [IN]    the storage format
 *
 * @retval ACL_SUCCESS    The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorFormat(aclTensorDesc *desc, aclFormat format);

/**
 * @ingroup AscendCL
 * @brief Set the shape specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  numDims [IN]    the number of dimensions of the shape
 * @param  dims [IN]       the size of the specified dimension
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorShape(aclTensorDesc *desc, int numDims, const int64_t *dims);

/**
 * @ingroup AscendCL
 * @brief Set the original format specified by the tensor description
 *
 * @param  desc [OUT]     pointer to the instance of aclTensorDesc
 * @param  format [IN]    the storage format
 *
 * @retval ACL_SUCCESS    The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorOriginFormat(aclTensorDesc *desc, aclFormat format);

/**
 * @ingroup AscendCL
 * @brief Set the original shape specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  numDims [IN]    the number of dimensions of the shape
 * @param  dims [IN]       the size of the specified dimension
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorOriginShape(aclTensorDesc *desc, int numDims, const int64_t *dims);

/**
 * @ingroup AscendCL
 * @brief get op description info
 *
 * @param desc [IN]     pointer to tensor description
 * @param index [IN]    index of tensor
 *
 * @retval null for failed.
 * @retval OtherValues success.
*/
ACL_FUNC_VISIBILITY aclTensorDesc *aclGetTensorDescByIndex(aclTensorDesc *desc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get address of tensor
 *
 * @param desc [IN]    pointer to tensor description
 *
 * @retval null for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY void *aclGetTensorDescAddress(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief Set the dynamic input name specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  dynamicInputName [IN]       pointer to the dynamic input name
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorDynamicInput(aclTensorDesc *desc, const char *dynamicInputName);

/**
 * @ingroup AscendCL
 * @brief Set const data specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  dataBuffer [IN]       pointer to the const databuffer
 * @param  length [IN]       the length of const databuffer
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorConst(aclTensorDesc *desc, void *dataBuffer, size_t length);

/**
 * @ingroup AscendCL
 * @brief Set tensor memory type specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  memType [IN]       ACL_MEMTYPE_DEVICE means device, ACL_MEMTYPE_HOST or
 * ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT means host
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorPlaceMent(aclTensorDesc *desc, aclMemType memType);

/**
 * @ingroup AscendCL
 * @brief an interface for users to output  APP logs
 *
 * @param logLevel [IN]    the level of current log
 * @param func [IN]        the function where the log is located
 * @param file [IN]        the file where the log is located
 * @param line [IN]        Number of source lines where the log is located
 * @param fmt [IN]         the format of current log
 * @param ... [IN]         the value of current log
 */
ACL_FUNC_VISIBILITY void aclAppLog(aclLogLevel logLevel, const char *func, const char *file, uint32_t line,
    const char *fmt, ...);

/**
 * @ingroup AscendCL
 * @brief get soc name
 *
 * @retval null for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY const char *aclrtGetSocName();

#define ACL_APP_LOG(level, fmt, ...) \
    aclAppLog(level, __FUNCTION__, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_BASE_H_


/**
* @file acl_rt.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_RT_H_
#define INC_EXTERNAL_ACL_ACL_RT_H_

#include <stdint.h>
#include <stddef.h>
// #include "acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_EVENT_SYNC                    0x00000001u
#define ACL_EVENT_CAPTURE_STREAM_PROGRESS 0x00000002u
#define ACL_EVENT_TIME_LINE               0x00000008u

#define ACL_STREAM_FAST_LAUNCH 0x00000001u
#define ACL_STREAM_FAST_SYNC   0x00000002u

#define ACL_CONTINUE_ON_FAILURE 0x00000000u
#define ACL_STOP_ON_FAILURE     0x00000001u

typedef enum aclrtRunMode {
    ACL_DEVICE,
    ACL_HOST,
} aclrtRunMode;

typedef enum aclrtTsId {
    ACL_TS_ID_AICORE   = 0,
    ACL_TS_ID_AIVECTOR = 1,
    ACL_TS_ID_RESERVED = 2,
} aclrtTsId;

typedef enum aclrtEventStatus {
    ACL_EVENT_STATUS_COMPLETE  = 0,
    ACL_EVENT_STATUS_NOT_READY = 1,
    ACL_EVENT_STATUS_RESERVED  = 2,
} aclrtEventStatus;

typedef enum aclrtEventRecordedStatus {
    ACL_EVENT_RECORDED_STATUS_NOT_READY = 0,
    ACL_EVENT_RECORDED_STATUS_COMPLETE = 1,
} aclrtEventRecordedStatus;

typedef enum aclrtEventWaitStatus {
    ACL_EVENT_WAIT_STATUS_COMPLETE  = 0,
    ACL_EVENT_WAIT_STATUS_NOT_READY = 1,
    ACL_EVENT_WAIT_STATUS_RESERVED  = 0xFFFF,
} aclrtEventWaitStatus;

typedef enum aclrtStreamStatus {
    ACL_STREAM_STATUS_COMPLETE  = 0,
    ACL_STREAM_STATUS_NOT_READY = 1,
    ACL_STREAM_STATUS_RESERVED  = 0xFFFF,
} aclrtStreamStatus;

typedef enum aclrtCallbackBlockType {
    ACL_CALLBACK_NO_BLOCK,
    ACL_CALLBACK_BLOCK,
} aclrtCallbackBlockType;

typedef enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

typedef enum aclrtMemMallocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST,
    ACL_MEM_MALLOC_HUGE_ONLY,
    ACL_MEM_MALLOC_NORMAL_ONLY,
    ACL_MEM_MALLOC_HUGE_FIRST_P2P,
    ACL_MEM_MALLOC_HUGE_ONLY_P2P,
    ACL_MEM_MALLOC_NORMAL_ONLY_P2P,
    ACL_MEM_TYPE_LOW_BAND_WIDTH   = 0x0100,
    ACL_MEM_TYPE_HIGH_BAND_WIDTH  = 0x1000,
} aclrtMemMallocPolicy;

typedef enum aclrtMemAttr {
    ACL_DDR_MEM,
    ACL_HBM_MEM,
    ACL_DDR_MEM_HUGE,
    ACL_DDR_MEM_NORMAL,
    ACL_HBM_MEM_HUGE,
    ACL_HBM_MEM_NORMAL,
    ACL_DDR_MEM_P2P_HUGE,
    ACL_DDR_MEM_P2P_NORMAL,
    ACL_HBM_MEM_P2P_HUGE,
    ACL_HBM_MEM_P2P_NORMAL,
} aclrtMemAttr;

typedef enum aclrtGroupAttr {
    ACL_GROUP_AICORE_INT,
    ACL_GROUP_AIV_INT,
    ACL_GROUP_AIC_INT,
    ACL_GROUP_SDMANUM_INT,
    ACL_GROUP_ASQNUM_INT,
    ACL_GROUP_GROUPID_INT
} aclrtGroupAttr;

typedef enum aclrtFloatOverflowMode {
    ACL_RT_OVERFLOW_MODE_SATURATION = 0,
    ACL_RT_OVERFLOW_MODE_INFNAN,
    ACL_RT_OVERFLOW_MODE_UNDEF,
} aclrtFloatOverflowMode;

typedef enum {
    ACL_RT_STREAM_WORK_ADDR_PTR = 0, /**< pointer to model work addr */
    ACL_RT_STREAM_WORK_SIZE, /**< pointer to model work size */
    ACL_RT_STREAM_FLAG,
    ACL_RT_STREAM_PRIORITY,
} aclrtStreamConfigAttr;

typedef struct aclrtStreamConfigHandle {
    void* workptr;
    size_t workSize;
    size_t flag;
    uint32_t priority;
} aclrtStreamConfigHandle;

typedef struct aclrtUtilizationExtendInfo aclrtUtilizationExtendInfo;

typedef struct aclrtUtilizationInfo {
    int32_t cubeUtilization;
    int32_t vectorUtilization;
    int32_t aicpuUtilization;
    int32_t memoryUtilization;
    aclrtUtilizationExtendInfo *utilizationExtend; /**< reserved parameters, current version needs to be null */
} aclrtUtilizationInfo;

typedef struct tagRtGroupInfo aclrtGroupInfo;

typedef struct rtExceptionInfo aclrtExceptionInfo;

typedef enum aclrtMemLocationType {
    ACL_MEM_LOCATION_TYPE_HOST = 0, /**< reserved enum, current version not support */
    ACL_MEM_LOCATION_TYPE_DEVICE,
} aclrtMemLocationType;

typedef struct aclrtMemLocation {
    uint32_t id;
    aclrtMemLocationType type;
} aclrtMemLocation;

typedef enum aclrtMemAllocationType {
    ACL_MEM_ALLOCATION_TYPE_PINNED = 0,
} aclrtMemAllocationType;

typedef enum aclrtMemHandleType {
    ACL_MEM_HANDLE_TYPE_NONE = 0,
} aclrtMemHandleType;

typedef struct aclrtPhysicalMemProp {
    aclrtMemHandleType handleType;
    aclrtMemAllocationType allocationType;
    aclrtMemAttr memAttr;
    aclrtMemLocation location;
    uint64_t reserve;
} aclrtPhysicalMemProp;

typedef void* aclrtDrvMemHandle;

typedef void (*aclrtCallback)(void *userData);

typedef void (*aclrtExceptionInfoCallback)(aclrtExceptionInfo *exceptionInfo);

typedef enum aclrtDeviceStatus {
    ACL_RT_DEVICE_STATUS_NORMAL = 0,
    ACL_RT_DEVICE_STATUS_ABNORMAL,
    ACL_RT_DEVICE_STATUS_END = 0xFFFF,
} aclrtDeviceStatus;

/**
 * @ingroup AscendCL
 * @brief Set a callback function to handle exception information
 *
 * @param callback [IN] callback function to handle exception information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback);

/**
 * @ingroup AscendCL
 * @brief Get task id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The task id from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetTaskIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get stream id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The stream id from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetStreamIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get thread id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The thread id of fail task
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetThreadIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get device id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The thread id of fail task
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetDeviceIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get error code from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The error code from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetErrorCodeFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief The thread that handles the callback function on the Stream
 *
 * @param threadId [IN] thread ID
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Add a callback function to be executed on the host
 *        to the task queue of the Stream
 *
 * @param fn [IN]   Specify the callback function to be added
 *                  The function prototype of the callback function is:
 *                  typedef void (*aclrtCallback)(void *userData);
 * @param userData [IN]   User data to be passed to the callback function
 * @param blockType [IN]  callback block type
 * @param stream [IN]     stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType,
                                                 aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief After waiting for a specified time, trigger callback processing
 *
 * @par Function
 *  The thread processing callback specified by
 *  the aclrtSubscribeReport interface
 *
 * @param timeout [IN]   timeout value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSubscribeReport
 */
ACL_FUNC_VISIBILITY aclError aclrtProcessReport(int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Cancel thread registration,
 *        the callback function on the specified Stream
 *        is no longer processed by the specified thread
 *
 * @param threadId [IN]   thread ID
 * @param stream [IN]     stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create context and associates it with the calling thread
 *
 * @par Function
 * The following use cases are supported:
 * @li If you don't call the aclrtCreateContext interface
 * to explicitly create the context,
 * the system will use the default context, which is implicitly created
 * when the aclrtSetDevice interface is called.
 * @li If multiple contexts are created in a process
 * (there is no limit on the number of contexts),
 * the current thread can only use one of them at the same time.
 * It is recommended to explicitly specify the context of the current thread
 * through the aclrtSetCurrentContext interface to increase.
 * the maintainability of the program.
 *
 * @param  context [OUT]    point to the created context
 * @param  deviceId [IN]    device to create context on
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSetDevice | aclrtSetCurrentContext
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief destroy context instance
 *
 * @par Function
 * Can only destroy context created through aclrtCreateContext interface
 *
 * @param  context [IN]   the context to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateContext
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyContext(aclrtContext context);

/**
 * @ingroup AscendCL
 * @brief set the context of the thread
 *
 * @par Function
 * The following scenarios are supported:
 * @li If the aclrtCreateContext interface is called in a thread to explicitly
 * create a Context (for example: ctx1), the thread's Context can be specified
 * without calling the aclrtSetCurrentContext interface.
 * The system uses ctx1 as the context of thread1 by default.
 * @li If the aclrtCreateContext interface is not explicitly created,
 * the system uses the default context as the context of the thread.
 * At this time, the aclrtDestroyContext interface cannot be used to release
 * the default context.
 * @li If the aclrtSetCurrentContext interface is called multiple times to
 * set the thread's Context, the last one prevails.
 *
 * @par Restriction
 * @li If the cevice corresponding to the context set for the thread
 * has been reset, you cannot set the context as the context of the thread,
 * otherwise a business exception will result.
 * @li It is recommended to use the context created in a thread.
 * If the aclrtCreateContext interface is called in thread A to create a context,
 * and the context is used in thread B,
 * the user must guarantee the execution order of tasks in the same stream
 * under the same context in two threads.
 *
 * @param  context [IN]   the current context of the thread
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateContext | aclrtDestroyContext
 */
ACL_FUNC_VISIBILITY aclError aclrtSetCurrentContext(aclrtContext context);

/**
 * @ingroup AscendCL
 * @brief get the context of the thread
 *
 * @par Function
 * If the user calls the aclrtSetCurrentContext interface
 * multiple times to set the context of the current thread,
 * then the last set context is obtained
 *
 * @param  context [OUT]   the current context of the thread
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSetCurrentContext
 */
ACL_FUNC_VISIBILITY aclError aclrtGetCurrentContext(aclrtContext *context);

/**
 * @ingroup AscendCL
 * @brief get system param option value in current context
 *
 * @param opt[IN] system option
 * @param value[OUT] value of system option
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtCtxGetSysParamOpt(aclSysParamOpt opt, int64_t *value);

/**
 * @ingroup AscendCL
 * @brief set system param option value in current context
 *
 * @param opt[IN] system option
 * @param value[IN] value of system option
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value);

/**
 * @ingroup AscendCL
 * @brief Specify the device to use for the operation
 * implicitly create the default context and the default stream
 *
 * @par Function
 * The following use cases are supported:
 * @li Device can be specified in the process or thread.
 * If you call the aclrtSetDevice interface multiple
 * times to specify the same device,
 * you only need to call the aclrtResetDevice interface to reset the device.
 * @li The same device can be specified for operation
 *  in different processes or threads.
 * @li Device is specified in a process,
 * and multiple threads in the process can share this device to explicitly
 * create a Context (aclrtCreateContext interface).
 * @li In multi-device scenarios, you can switch to other devices
 * through the aclrtSetDevice interface in the process.
 *
 * @param  deviceId [IN]  the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtResetDevice |aclrtCreateContext
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDevice(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief Reset the current operating Device and free resources on the device,
 * including the default context, the default stream,
 * and all streams created under the default context,
 * and synchronizes the interface.
 * If the task under the default context or stream has not been completed,
 * the system will wait for the task to complete before releasing it.
 *
 * @par Restriction
 * @li The Context, Stream, and Event that are explicitly created
 * on the device to be reset. Before resetting,
 * it is recommended to follow the following interface calling sequence,
 * otherwise business abnormalities may be caused.
 * @li Interface calling sequence:
 * call aclrtDestroyEvent interface to release Event or
 * call aclrtDestroyStream interface to release explicitly created Stream->
 * call aclrtDestroyContext to release explicitly created Context->
 * call aclrtResetDevice interface
 *
 * @param  deviceId [IN]   the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetDevice(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief get target device of current thread
 *
 * @param deviceId [OUT]  the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDevice(int32_t *deviceId);

/**
 * @ingroup AscendCL
 * @brief set stream failure mode
 *
 * @param stream [IN]  the stream to set
 * @param mode [IN]  stream failure mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode);

/**
 * @ingroup AscendCL
 * @brief get target side
 *
 * @param runMode [OUT]    the run mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetRunMode(aclrtRunMode *runMode);

/**
 * @ingroup AscendCL
 * @brief Wait for compute device to finish
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeDevice(void);

/**
 * @ingroup AscendCL
 * @brief Set Scheduling TS
 *
 * @param tsId [IN]   the ts id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetTsDevice(aclrtTsId tsId);

/**
 * @ingroup AscendCL
 * @brief Query the comprehensive usage rate of device
 * @param deviceId [IN] the need query's deviceId
 * @param utilizationInfo [OUT] the usage rate of device
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo *utilizationInfo);

/**
 * @ingroup AscendCL
 * @brief get total device number.
 *
 * @param count [OUT]    the device number
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceCount(uint32_t *count);

/**
 * @ingroup AscendCL
 * @brief create event instance
 *
 * @param event [OUT]   created event
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateEvent(aclrtEvent *event);

/**
 * @ingroup AscendCL
 * @brief create event instance with flag
 *
 * @param event [OUT]   created event
 * @param flag [IN]     event flag
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief destroy event instance
 *
 * @par Function
 *  Only events created through the aclrtCreateEvent interface can be
 *  destroyed, synchronous interfaces. When destroying an event,
 *  the user must ensure that the tasks involved in the aclrtSynchronizeEvent
 *  interface or the aclrtStreamWaitEvent interface are completed before
 *  they are destroyed.
 *
 * @param  event [IN]   event to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateEvent | aclrtSynchronizeEvent | aclrtStreamWaitEvent
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyEvent(aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief Record an Event in the Stream
 *
 * @param event [IN]    event to record
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Reset an event
 *
 * @par Function
 *  Users need to make sure to wait for the tasks in the Stream
 *  to complete before resetting the Event
 *
 * @param event [IN]    event to reset
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream);

 /**
 * @ingroup AscendCL
 * @brief Queries an event's status
 *
 * @param  event [IN]    event to query
 * @param  status [OUT]  event status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_DEPRECATED_MESSAGE("aclrtQueryEvent is deprecated, use aclrtQueryEventStatus instead")
ACL_FUNC_VISIBILITY aclError aclrtQueryEvent(aclrtEvent event, aclrtEventStatus *status);

/**
 * @ingroup AscendCL
 * @brief Queries an event's status
 *
 * @param  event [IN]    event to query
 * @param  status [OUT]  event recorded status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status);

/**
* @ingroup AscendCL
* @brief Queries an event's wait-status
*
* @param  event [IN]    event to query
* @param  status [OUT]  event wait-status
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus *status);

/**
 * @ingroup AscendCL
 * @brief Block Host Running, wait event to be complete
 *
 * @param  event [IN]   event to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeEvent(aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief Block Host Running, wait event to be complete
 *
 * @param  event [IN]   event to wait
 * @param  timeout [IN]  timeout value,the unit is milliseconds
 * -1 means waiting indefinitely, 0 means check whether synchronization is immediately completed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeEventWithTimeout(aclrtEvent event, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief computes the elapsed time between events.
 *
 * @param ms [OUT]     time between start and end in ms
 * @param start [IN]   starting event
 * @param end [IN]     ending event
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateEvent | aclrtRecordEvent | aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent);

/**
 * @ingroup AscendCL
 * @brief alloc memory on device, real alloc size is aligned to 32 bytes and padded with 32 bytes
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMalloc interface needs to be released
 * through the aclrtFree interface.
 * @li Before calling the media data processing interface,
 * if you need to apply memory on the device to store input or output data,
 * you need to call acldvppMalloc to apply for memory.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | acldvppMalloc | aclrtMallocCached
 */
ACL_FUNC_VISIBILITY aclError aclrtMalloc(void **devPtr,
                                         size_t size,
                                         aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief alloc memory on device, real alloc size is aligned to 32 bytes with no padding
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMallocAlign32 interface needs to be released
 * through the aclrtFree interface.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | aclrtMalloc | aclrtMallocCached
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocAlign32(void **devPtr,
                                                size_t size,
                                                aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief allocate memory on device with cache
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMallocCached interface needs to be released
 * through the aclrtFree interface.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | aclrtMalloc
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocCached(void **devPtr,
                                               size_t size,
                                               aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief flush cache data to ddr
 *
 * @param devPtr [IN]  the pointer that flush data to ddr
 * @param size [IN]    flush size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemFlush(void *devPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief invalidate cache data
 *
 * @param devPtr [IN]  pointer to invalidate cache data
 * @param size [IN]    invalidate size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemInvalidate(void *devPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief free device memory
 *
 * @par Function
 *  can only free memory allocated through the aclrtMalloc interface
 *
 * @param  devPtr [IN]  Pointer to memory to be freed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMalloc
 */
ACL_FUNC_VISIBILITY aclError aclrtFree(void *devPtr);

/**
 * @ingroup AscendCL
 * @brief alloc memory on host
 *
 * @par Restriction
 * @li The requested memory cannot be used in the Device
 * and needs to be explicitly copied to the Device.
 * @li The memory requested by the aclrtMallocHost interface
 * needs to be released through the aclrtFreeHost interface.
 *
 * @param  hostPtr [OUT] pointer to pointer to allocated memory on the host
 * @param  size [IN]     alloc memory size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFreeHost
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocHost(void **hostPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief free host memory
 *
 * @par Function
 *  can only free memory allocated through the aclrtMallocHost interface
 *
 * @param  hostPtr [IN]   free memory pointer
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMallocHost
 */
ACL_FUNC_VISIBILITY aclError aclrtFreeHost(void *hostPtr);

/**
 * @ingroup AscendCL
 * @brief synchronous memory replication between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param destMax [IN]   Max length of the destination address memory
 * @param src [IN]       source address pointer
 * @param count [IN]     the length of byte to copy
 * @param kind [IN]      memcpy type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy(void *dst,
                                         size_t destMax,
                                         const void *src,
                                         size_t count,
                                         aclrtMemcpyKind kind);

/**
 * @ingroup AscendCL
 * @brief Initialize memory and set contents of memory to specified value
 *
 * @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
 * @param devPtr [IN]    Starting address of memory
 * @param maxCount [IN]  Max length of destination address memory
 * @param value [IN]     Set value
 * @param count [IN]     The length of memory
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);

/**
 * @ingroup AscendCL
 * @brief  Asynchronous memory replication between Host and Device
 *
 * @par Function
 *  After calling this interface,
 *  be sure to call the aclrtSynchronizeStream interface to ensure that
 *  the task of memory replication has been completed
 *
 * @par Restriction
 * @li For on-chip Device-to-Device memory copy,
 *     both the source and destination addresses must be 64-byte aligned
 *
 * @param dst [IN]     destination address pointer
 * @param destMax [IN] Max length of destination address memory
 * @param src [IN]     source address pointer
 * @param count [IN]   the number of byte to copy
 * @param kind [IN]    memcpy type
 * @param stream [IN]  asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsync(void *dst,
                                              size_t destMax,
                                              const void *src,
                                              size_t count,
                                              aclrtMemcpyKind kind,
                                              aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief synchronous memory replication of two-dimensional matrix between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param dpitch [IN]    pitch of destination memory
 * @param src [IN]       source address pointer
 * @param spitch [IN]    pitch of source memory
 * @param width [IN]     width of matrix transfer
 * @param height [IN]    height of matrix transfer
 * @param kind [IN]      memcpy type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy2d(void *dst,
                                           size_t dpitch,
                                           const void *src,
                                           size_t spitch,
                                           size_t width,
                                           size_t height,
                                           aclrtMemcpyKind kind);

/**
 * @ingroup AscendCL
 * @brief asynchronous memory replication of two-dimensional matrix between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param dpitch [IN]    pitch of destination memory
 * @param src [IN]       source address pointer
 * @param spitch [IN]    pitch of source memory
 * @param width [IN]     width of matrix transfer
 * @param height [IN]    height of matrix transfer
 * @param kind [IN]      memcpy type
 * @param stream [IN]    asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy2dAsync(void *dst,
                                                size_t dpitch,
                                                const void *src,
                                                size_t spitch,
                                                size_t width,
                                                size_t height,
                                                aclrtMemcpyKind kind,
                                                aclrtStream stream);

/**
* @ingroup AscendCL
* @brief Asynchronous initialize memory
* and set contents of memory to specified value async
*
* @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
* @param devPtr [IN]      destination address pointer
* @param maxCount [IN]    Max length of destination address memory
* @param value [IN]       set value
* @param count [IN]       the number of byte to set
* @param stream [IN]      asynchronized task stream
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*
* @see aclrtSynchronizeStream
*/
ACL_FUNC_VISIBILITY aclError aclrtMemsetAsync(void *devPtr,
                                              size_t maxCount,
                                              int32_t value,
                                              size_t count,
                                              aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Allocate an address range reservation
 *
 * @param virPtr [OUT]    Resulting pointer to start of virtual address range allocated
 * @param size [IN]       Size of the reserved virtual address range requested
 * @param alignment [IN]  Alignment of the reserved virtual address range requested
 * @param expectPtr [IN]  Fixed starting address range requested, must be nullptr
 * @param flags [IN]      Flag of page type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtReleaseMemAddress | aclrtMallocPhysical | aclrtMapMem
 */
ACL_FUNC_VISIBILITY aclError aclrtReserveMemAddress(void **virPtr,
                                                    size_t size,
                                                    size_t alignment,
                                                    void *expectPtr,
                                                    uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Free an address range reservation
 *
 * @param virPtr [IN]  Starting address of the virtual address range to free
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtReserveMemAddress
 */
ACL_FUNC_VISIBILITY aclError aclrtReleaseMemAddress(void *virPtr);

/**
 * @ingroup AscendCL
 * @brief Create a memory handle representing a memory allocation of a given
 * size described by the given properties
 *
 * @param handle [OUT]  Value of handle returned. All operations on this
 * allocation are to be performed using this handle.
 * @param size [IN]     Size of the allocation requested
 * @param prop [IN]     Properties of the allocation to create
 * @param flags [IN]    Currently unused, must be zero
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFreePhysical | aclrtReserveMemAddress | aclrtMapMem
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle,
                                                 size_t size,
                                                 const aclrtPhysicalMemProp *prop,
                                                 uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Release a memory handle representing a memory allocation which was
 * previously allocated through aclrtMallocPhysical
 *
 * @param handle [IN]  Value of handle which was returned previously by aclrtMallocPhysical
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMallocPhysical
 */
ACL_FUNC_VISIBILITY aclError aclrtFreePhysical(aclrtDrvMemHandle handle);

/**
 * @ingroup AscendCL
 * @brief Maps an allocation handle to a reserved virtual address range
 *
 * @param virPtr [IN]  Address where memory will be mapped
 * @param size [IN]    Size of the memory mapping
 * @param offset [IN]  Offset into the memory represented by handle from which to start mapping
 * @param handle [IN]  Handle to a shareable memory
 * @param flags [IN]   Currently unused, must be zero
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtUnmapMem | aclrtReserveMemAddress | aclrtMallocPhysical
 */
ACL_FUNC_VISIBILITY aclError aclrtMapMem(void *virPtr,
                                         size_t size,
                                         size_t offset,
                                         aclrtDrvMemHandle handle,
                                         uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Unmap the backing memory of a given address range
 *
 * @param virPtr [IN]  Starting address for the virtual address range to unmap
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMapMem
 */
ACL_FUNC_VISIBILITY aclError aclrtUnmapMem(void *virPtr);

/**
 * @ingroup AscendCL
 * @brief Create config handle of stream
 *
 * @retval the aclrtStreamConfigHandle pointer
 */
ACL_FUNC_VISIBILITY aclrtStreamConfigHandle *aclrtCreateStreamConfigHandle(void);

/**
 * @ingroup AscendCL
 * @brief Destroy config handle of model execute
 *
 * @param  handle [IN]  Pointer to aclrtStreamConfigHandle to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyStreamConfigHandle(aclrtStreamConfigHandle *handle);

/**
 * @ingroup AscendCL
 * @brief set config for stream
 *
 * @param handle [OUT]    pointer to stream config handle
 * @param attr [IN]       config attr in stream config handle to be set
 * @param attrValue [IN]  pointer to stream config value
 * @param valueSize [IN]  memory size of attrValue
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamConfigOpt(aclrtStreamConfigHandle *handle, aclrtStreamConfigAttr attr,
    const void *attrValue, size_t valueSize);

/**
 * @ingroup AscendCL
 * @brief  create stream instance
 *
 * @param  stream [OUT]   the created stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateStream(aclrtStream *stream);

/**
 * @ingroup AscendCL
 * @brief  create stream instance
 *
 * @param  stream [OUT]   the created stream
 * @param  handle [IN]   the config of stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateStreamV2(aclrtStream *stream, const aclrtStreamConfigHandle *handle);

/**
 * @ingroup AscendCL
 * @brief  create stream instance with param
 *
 * @par Function
 * Can create fast streams through the aclrtCreateStreamWithConfig interface
 *
 * @param  stream [OUT]   the created stream
 * @param  priority [IN]   the priority of stream, value range:0~7
 * @param  flag [IN]   indicate the function for stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief destroy stream instance
 *
 * @par Function
 * Can only destroy streams created through the aclrtCreateStream interface
 *
 * @par Restriction
 * Before calling the aclrtDestroyStream interface to destroy
 * the specified Stream, you need to call the aclrtSynchronizeStream interface
 * to ensure that the tasks in the Stream have been completed.
 *
 * @param stream [IN]  the stream to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateStream | aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyStream(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief destroy stream instance by force
 *
 * @par Function
 * Can only destroy streams created through the aclrtCreateStream interface
 *
 * @param stream [IN]  the stream to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateStream
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyStreamForce(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief block the host until all tasks
 * in the specified stream have completed
 *
 * @param  stream [IN]   the stream to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStream(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief block the host until all tasks
 * in the specified stream have completed
 *
 * @param  stream [IN]   the stream to wait
 * @param  timeout [IN]  timeout value,the unit is milliseconds
 * -1 means waiting indefinitely, 0 means check whether synchronization is complete immediately
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Query a stream for completion status.
 *
 * @param  stream [IN]   the stream to query
 * @param  status [OUT]  stream status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status);

/**
 * @ingroup AscendCL
 * @brief Blocks the operation of the specified Stream until
 * the specified Event is completed.
 * Support for multiple streams waiting for the same event.
 *
 * @param  stream [IN]   the wait stream If using thedefault Stream, set NULL
 * @param  event [IN]    the event to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief set group
 *
 * @par Function
 *  set the task to the corresponding group
 *
 * @param groupId [IN]   group id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount | aclrtGetAllGroupInfo | aclrtGetGroupInfoDetail
 */
ACL_FUNC_VISIBILITY aclError aclrtSetGroup(int32_t groupId);

/**
 * @ingroup AscendCL
 * @brief get the number of group
 *
 * @par Function
 *  get the number of group. if the number of group is zero,
 *  it means that group is not supported or group is not created.
 *
 * @param count [OUT]   the number of group
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 */
ACL_FUNC_VISIBILITY aclError aclrtGetGroupCount(uint32_t *count);

/**
 * @ingroup AscendCL
 * @brief create group information
 *
 * @retval null for failed.
 * @retval OtherValues success.
 *
 * @see aclrtDestroyGroupInfo
 */
ACL_FUNC_VISIBILITY aclrtGroupInfo *aclrtCreateGroupInfo();

/**
 * @ingroup AscendCL
 * @brief destroy group information
 *
 * @param groupInfo [IN]   pointer to group information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateGroupInfo
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyGroupInfo(aclrtGroupInfo *groupInfo);

/**
 * @ingroup AscendCL
 * @brief get all group information
 *
 * @param groupInfo [OUT]   pointer to group information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount
 */
ACL_FUNC_VISIBILITY aclError aclrtGetAllGroupInfo(aclrtGroupInfo *groupInfo);

/**
 * @ingroup AscendCL
 * @brief get detail information of group
 *
 * @param groupInfo [IN]    pointer to group information
 * @param groupIndex [IN]   group index value
 * @param attr [IN]         group attribute
 * @param attrValue [OUT]   pointer to attribute value
 * @param valueLen [IN]     length of attribute value
 * @param paramRetSize [OUT]   pointer to real length of attribute value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount | aclrtGetAllGroupInfo
 */
ACL_FUNC_VISIBILITY aclError aclrtGetGroupInfoDetail(const aclrtGroupInfo *groupInfo,
                                                     int32_t groupIndex,
                                                     aclrtGroupAttr attr,
                                                     void *attrValue,
                                                     size_t valueLen,
                                                     size_t *paramRetSize);

/**
 * @ingroup AscendCL
 * @brief checking whether current device and peer device support the p2p feature
 *
 * @param canAccessPeer [OUT]   pointer to save the checking result
 * @param deviceId [IN]         current device id
 * @param peerDeviceId [IN]     peer device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceEnablePeerAccess | aclrtDeviceDisablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId);

/**
 * @ingroup AscendCL
 * @brief enable the peer device to support the p2p feature
 *
 * @param peerDeviceId [IN]   the peer device id
 * @param flags [IN]   reserved field, now it must be zero
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceCanAccessPeer | aclrtDeviceDisablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags);

/**
 * @ingroup AscendCL
 * @brief disable the peer device to support the p2p function
 *
 * @param peerDeviceId [IN]   the peer device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceCanAccessPeer | aclrtDeviceEnablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceDisablePeerAccess(int32_t peerDeviceId);

/**
 * @ingroup AscendCL
 * @brief Obtain the free memory and total memory of specified attribute.
 * the specified memory include normal memory and huge memory.
 *
 * @param attr [IN]    the memory attribute of specified device
 * @param free [OUT]   the free memory of specified device
 * @param total [OUT]  the total memory of specified device.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total);

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for waitting of op
 *
 * @param timeout [IN]   op wait timeout
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpWaitTimeout(uint32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for op executing
 *
 * @param timeout [IN]   op execute timeout
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpExecuteTimeOut(uint32_t timeout);

/**
 * @ingroup AscendCL
 * @brief enable or disable overflow switch on some stream
 * @param stream [IN]   set overflow switch on this stream
 * @param flag [IN]  0 : disable 1 : enable
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief get overflow switch on some stream
 * @param stream [IN]   get overflow switch on this stream
 * @param flag [OUT]  current overflow switch, 0 : disable others : enable
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetStreamOverflowSwitch(aclrtStream stream, uint32_t *flag);

/**
 * @ingroup AscendCL
 * @brief set saturation mode
 * @param mode [IN]   target saturation mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode);

/**
 * @ingroup AscendCL
 * @brief get saturation mode
 * @param mode [OUT]   get saturation mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceSatMode(aclrtFloatOverflowMode *mode);

/**
 * @ingroup AscendCL
 * @brief get overflow status asynchronously
 *
 * @par Restriction
 * After calling the aclrtGetOverflowStatus interface,
 * you need to call the aclrtSynchronizeStream interface
 * to ensure that the tasks in the stream have been completed.
 * @param outputAddr [IN/OUT]  output device addr to store overflow status
 * @param outputSize [IN]  output addr size
 * @param outputSize [IN]  stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetOverflowStatus(void *outputAddr, size_t outputSize, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief reset overflow status asynchronously
 *
 * @par Restriction
 * After calling the aclrtResetOverflowStatus interface,
 * you need to call the aclrtSynchronizeStream interface
 * to ensure that the tasks in the stream have been completed.
 * @param outputSize [IN]  stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetOverflowStatus(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief The thread that handles the hostFunc function on the Stream
 *
 * @param hostFuncThreadId [IN] thread ID
 * @param exeStream        [IN] stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSubscribeHostFunc(uint64_t hostFuncThreadId, aclrtStream exeStream);

/**
 * @ingroup AscendCL
 * @brief After waiting for a specified time, trigger hostFunc callback function processing
 *
 * @par Function
 *  The thread processing callback specified by the aclrtSubscribeHostFunc interface
 *
 * @param timeout [IN]   timeout value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSubscribeHostFunc
 */
ACL_FUNC_VISIBILITY aclError aclrtProcessHostFunc(int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Cancel thread registration,
 *        the hostFunc function on the specified Stream
 *        is no longer processed by the specified thread
 *
 * @param hostFuncThreadId [IN]   thread ID
 * @param exeStream        [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtUnSubscribeHostFunc(uint64_t hostFuncThreadId, aclrtStream exeStream);

/**
 * @ingroup AscendCL
 * @brief Get device status
 *
 * @param deviceId       [IN]   device ID
 * @param deviceStatus   [OUT]  device status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtQueryDeviceStatus(int32_t deviceId, aclrtDeviceStatus *deviceStatus);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_RT_H_


/**
* @file acl_op.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef INC_EXTERNAL_ACL_ACL_OP_H_
#define INC_EXTERNAL_ACL_ACL_OP_H_

// #include "acl_base.h"
// #include "acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aclopHandle aclopHandle;
typedef struct aclopAttr aclopAttr;
typedef struct aclopKernelDesc aclopKernelDesc;

typedef void (*aclDataDeallocator)(void *data, size_t length);

static const int ACL_COMPILE_FLAG_BIN_SELECTOR = 1;

typedef enum aclEngineType {
    ACL_ENGINE_SYS,
    ACL_ENGINE_AICORE,
    ACL_ENGINE_VECTOR,
} aclopEngineType;

/**
 * @ingroup AscendCL
 * @brief Set base directory that contains single op models
 *
 * @par Restriction
 * The aclopSetModelDir interface can be called only once in a process.
 * @param  modelDir [IN]   path of the directory
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetModelDir(const char *modelDir);

/**
 * @ingroup AscendCL
 * @brief load single op models from memory
 *
 * @par Restriction
 * The aclopLoad interface can be called more than one times in a process.
 * @param model [IN]        address of single op models
 * @param modelSize [IN]    size of single op models
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopLoad(const void *model, size_t modelSize);

/**
 * @ingroup AscendCL
 * @brief create data of type aclopAttr
 *
 * @retval pointer to created instance.
 * @retval nullptr if run out of memory
 */
ACL_FUNC_VISIBILITY aclopAttr *aclopCreateAttr();

/**
 * @ingroup AscendCL
 * @brief destroy data of typ aclopAttr
 *
 * @param attr [IN]   pointer to the instance of aclopAttr
 */
ACL_FUNC_VISIBILITY void aclopDestroyAttr(const aclopAttr *attr);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is bool
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *                         false if attrValue is 0, true otherwise.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is int64_t
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrInt(aclopAttr *attr, const char *attrName, int64_t attrValue);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is float
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrFloat(aclopAttr *attr, const char *attrName, float attrValue);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is string
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrString(aclopAttr *attr, const char *attrName, const char *attrValue);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is aclDataType
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrDataType(aclopAttr *attr, const char *attrName, aclDataType attrValue);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of aclDataType
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values. false if attrValue is 0, true otherwise.
 * @param values [IN]      pointer to values
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListDataType(aclopAttr *attr, const char *attrName, int numValues,
    const aclDataType values[]);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of bools
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values. false if attrValue is 0, true otherwise.
 * @param values [IN]      pointer to values
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListBool(aclopAttr *attr, const char *attrName, int numValues,
    const uint8_t *values);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of ints
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values
 * @param values [IN]      pointer to values
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListInt(aclopAttr *attr, const char *attrName, int numValues,
    const int64_t *values);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of floats
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values
 * @param values [IN]      pointer to values
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListFloat(aclopAttr *attr, const char *attrName, int numValues,
    const float *values);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of strings
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values
 * @param values [IN]      pointer to values
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListString(aclopAttr *attr, const char *attrName, int numValues,
    const char **values);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of list of ints
 *
 * @param attr [OUT]       pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numLists [IN]    number of lists
 * @param numValues [IN]   pointer to number of values of each list
 * @param values [IN]      pointer to values
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListListInt(aclopAttr *attr,
                                                     const char *attrName,
                                                     int numLists,
                                                     const int *numValues,
                                                     const int64_t *const values[]);

/**
 * @ingroup AscendCL
 * @brief Load and execute the specified operator asynchronously
 *
 * @par Restriction
 * @li The input and output organization of each operator is different,
 * and the application needs to organize the operator strictly
 * according to the operator input and output parameters when calling.
 * @li When the user calls aclopExecute,
 * the ACL finds the corresponding task according to the optype,
 * the description of the input tesnsor,
 * the description of the output tesnsor, and attr, and issues the execution.
 *
 * @param opType [IN]      type of op
 * @param numInputs [IN]   number of inputs
 * @param inputDesc [IN]   pointer to array of input tensor descriptions
 * @param inputs [IN]      pointer to array of input buffers
 * @param numOutputs [IN]  number of outputs
 * @param outputDesc [IN]  pointer to array of output tensor descriptions
 * @param outputs [OUT]    pointer to array of output buffers
 * @param attr [IN]        pointer to instance of aclopAttr.
 *                         may pass nullptr if the op has no attribute
 * @param stream [IN]      stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_DEPRECATED_MESSAGE("aclopExecute is deprecated, use aclopExecuteV2 instead")
ACL_FUNC_VISIBILITY aclError aclopExecute(const char *opType,
                                          int numInputs,
                                          const aclTensorDesc *const inputDesc[],
                                          const aclDataBuffer *const inputs[],
                                          int numOutputs,
                                          const aclTensorDesc *const outputDesc[],
                                          aclDataBuffer *const outputs[],
                                          const aclopAttr *attr,
                                          aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Load and execute the specified operator
 *        The difference with aclopExecute is that aclopExecuteV2 will refresh outputDesc
 *
 * @par Restriction
 * @li The input and output organization of each operator is different,
 * and the application needs to organize the operator strictly
 * according to the operator input and output parameters when calling.
 * @li When the user calls aclopExecuteV2,
 * the ACL finds the corresponding task according to the optype,
 * the description of the input tesnsor,
 * the description of the output tesnsor, and attr, and issues the execution.
 *
 * @param opType [IN]      type of op
 * @param numInputs [IN]   number of inputs
 * @param inputDesc [IN]   pointer to array of input tensor descriptions
 * @param inputs [IN]      pointer to array of input buffers
 * @param numOutputs [IN]  number of outputs
 * @param outputDesc [IN|OUT]  pointer to array of output tensor descriptions
 * @param outputs [OUT]    pointer to array of output buffers
 * @param attr [IN]        pointer to instance of aclopAttr.
 *                         may pass nullptr if the op has no attribute
 * @param stream [IN]      stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopExecuteV2(const char *opType,
                                            int numInputs,
                                            aclTensorDesc *inputDesc[],
                                            aclDataBuffer *inputs[],
                                            int numOutputs,
                                            aclTensorDesc *outputDesc[],
                                            aclDataBuffer *outputs[],
                                            aclopAttr *attr,
                                            aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create a instance of aclopHandle.
 *
 * @param opType [IN]      type of op
 * @param numInputs [IN]   number of inputs
 * @param inputDesc [IN]   pointer to array of input tensor descriptions
 * @param numOutputs [IN]  number of outputs
 * @param outputDesc [IN]  pointer to array of output tensor descriptions
 * @param opAttr [IN]      pointer to instance of aclopAttr.
 *                         may pass nullptr if the op has no attribute
 * @param handle [OUT]     pointer to the pointer to the handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCreateHandle(const char *opType,
                                               int numInputs,
                                               const aclTensorDesc *const inputDesc[],
                                               int numOutputs,
                                               const aclTensorDesc *const outputDesc[],
                                               const aclopAttr *opAttr,
                                               aclopHandle **handle);

/**
 * @ingroup AscendCL
 * @brief destroy aclopHandle instance
 *
 * @param handle [IN]   pointer to the instance of aclopHandle
 */
ACL_FUNC_VISIBILITY void aclopDestroyHandle(aclopHandle *handle);

/**
 * @ingroup AscendCL
 * @brief execute an op with the handle.
 *        can save op model matching cost compared with aclopExecute
 *
 * @param handle [IN]      pointer to the instance of aclopHandle.
 *                         The aclopCreateHandle interface has been called
 *                         in advance to create aclopHandle type data.
 * @param numInputs [IN]   number of inputs
 * @param inputs [IN]      pointer to array of input buffers.
 *                         The aclCreateDataBuffer interface has been called
 *                         in advance to create aclDataBuffer type data.
 * @param numOutputs [IN]  number of outputs
 * @param outputs [OUT]    pointer to array of output buffers
 * @param stream [IN]      stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclopCreateHandle | aclCreateDataBuffer
 */
ACL_FUNC_VISIBILITY aclError aclopExecWithHandle(aclopHandle *handle,
                                                 int numInputs,
                                                 const aclDataBuffer *const inputs[],
                                                 int numOutputs,
                                                 aclDataBuffer *const outputs[],
                                                 aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief cast data type
 *
 * @param srcDesc [IN]     source tensor desc
 * @param srcBuffer [IN]   source tensor buffer
 * @param dstDesc [IN]     destination tensor desc
 * @param dstBuffer [OUT]  destination tensor buffer
 * @param truncate [IN]    do not truncate if value is 0, truncate otherwise
 * @param stream [IN]      stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCast(const aclTensorDesc *srcDesc,
                                       const aclDataBuffer *srcBuffer,
                                       const aclTensorDesc *dstDesc,
                                       aclDataBuffer *dstBuffer,
                                       uint8_t truncate,
                                       aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create a handle for casting datatype
 *
 * @param srcDesc [IN]    source tensor desc
 * @param dstDesc [IN]    destination tensor desc
 * @param truncate [IN]   do not truncate if value is 0, truncate otherwise
 * @param handle [OUT]    pointer to the pointer to the handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCreateHandleForCast(aclTensorDesc *srcDesc,
                                                      aclTensorDesc *dstDesc,
                                                      uint8_t truncate,
                                                      aclopHandle **handle);


/**
 * @ingroup AscendCL
 * @brief create kernel
 *
 * @param opType [IN]           op type
 * @param kernelId [IN]         kernel id
 * @param kernelName [IN]       kernel name
 * @param binData [IN]          kernel bin data
 * @param binSize [IN]          kernel bin size
 * @param enginetype [IN]       enigne type
 * @param deallocator [IN]      callback function for deallocating bin data,
 *                              null if bin data to be deallocated by caller
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclopCompile
 */
ACL_FUNC_VISIBILITY aclError aclopCreateKernel(const char *opType,
                                               const char *kernelId,
                                               const char *kernelName,
                                               void *binData,
                                               int binSize,
                                               aclopEngineType enginetype,
                                               aclDataDeallocator deallocator);


/**
 * @ingroup AscendCL
 * @brief create kernel
 *
 * @param numInputs [IN]            number of inputs
 * @param inputDesc [IN]            pointer to array of input tensor descriptions
 * @param numOutputs [IN]           number of outputs
 * @param outputDesc [IN]           pointer to array of output tensor descriptions
 * @param opAttr [IN]               pointer to instance of aclopAttr
 * @param aclopKernelDesc [IN]      pointer to instance of aclopKernelDesc
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
typedef aclError (*aclopCompileFunc)(int numInputs,
                                     const aclTensorDesc *const inputDesc[],
                                     int numOutputs,
                                     const aclTensorDesc *const outputDesc[],
                                     const aclopAttr *opAttr,
                                     aclopKernelDesc *aclopKernelDesc);

/**
 * @ingroup AscendCL
 * @brief register compile function
 *
 * @param opType [IN]         op type
 * @param func [IN]           compile function
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclopUnregisterCompileFunc
 */
ACL_FUNC_VISIBILITY aclError aclopRegisterCompileFunc(const char *opType, aclopCompileFunc func);

/**
 * @ingroup AscendCL
 * @brief unregister compile function
 *
 * @param opType [IN]         op type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopUnregisterCompileFunc(const char *opType);

/**
 * @ingroup AscendCL
 * @brief set kernel args
 *
 * @param kernelDesc [IN]               pointer to instance of aclopKernelDesc
 * @param kernelId [IN]                 kernel id
 * @param blockDim [IN]                 block dim
 * @param args [IN]                     args
 * @param argSize [IN]                  size in bytes of args
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetKernelArgs(aclopKernelDesc *kernelDesc,
                                                const char *kernelId,
                                                uint32_t blockDim,
                                                const void *args,
                                                uint32_t argSize);

/**
 * @ingroup AscendCL
 * @brief set workspace sizes
 *
 * @param kernelDesc [IN]               pointer to instance of aclopKernelDesc
 * @param numWorkspaces [IN]            number of workspaces
 * @param workspaceSizes [IN]           pointer to array of sizes of workspaces
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetKernelWorkspaceSizes(aclopKernelDesc *kernelDesc, int numWorkspaces,
                                                          size_t *workspaceSizes);

/**
 * @ingroup AscendCL
 * @brief compile op with dynamic shape
 *
 * @param opType [IN]       op type
 * @param numInputs [IN]    number of inputs
 * @param inputDesc [IN]    pointer to array of input tensor descriptions
 * @param numOutputs [IN]   number of outputs
 * @param outputDesc [IN]   pointer to array of output tensor descriptions
 * @param attr [IN]         pointer to instance of aclopAttr.
 *                          may pass nullptr if the op has no attribute
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopUpdateParams(const char *opType,
                                               int numInputs,
                                               const aclTensorDesc *const inputDesc[],
                                               int numOutputs,
                                               const aclTensorDesc *const outputDesc[],
                                               const aclopAttr *attr);

/**
 * @ingroup AscendCL
 * @brief inferShape the specified operator synchronously
 *
 * @param opType [IN]       type of op
 * @param numInputs [IN]    number of inputs
 * @param inputDesc [IN]    pointer to array of input tensor descriptions
 * @param inputs [IN]       pointer to array of input buffers
 * @param numOutputs [IN]   number of outputs
 * @param outputDesc [OUT]  pointer to array of output tensor descriptions
 * @param attr [IN]         pointer to instance of aclopAttr.
 *                          may pass nullptr if the op has no attribute
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopInferShape(const char *opType,
                                             int numInputs,
                                             aclTensorDesc *inputDesc[],
                                             aclDataBuffer *inputs[],
                                             int numOutputs,
                                             aclTensorDesc *outputDesc[],
                                             aclopAttr *attr);

#define ACL_OP_DUMP_OP_AICORE_ARGS 0x00000001U

/**
 * @ingroup AscendCL
 * @brief Enable the dump function of the corresponding dump type.
 *
 * @param dumpType [IN]       type of dump
 * @param path     [IN]       dump path
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopStartDumpArgs(uint32_t dumpType, const char *path);

/**
 * @ingroup AscendCL
 * @brief Disable the dump function of the corresponding dump type.
 *
 * @param dumpType [IN]       type of dump
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopStopDumpArgs(uint32_t dumpType);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_OP_H_


/**
* @file acl_mdl.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_MODEL_H_
#define INC_EXTERNAL_ACL_ACL_MODEL_H_

#include <stddef.h>
#include <stdint.h>

// #include "acl_base.h"
// #include "acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_MAX_DIM_CNT          128
#define ACL_MAX_TENSOR_NAME_LEN  128
#define ACL_MAX_BATCH_NUM        128
#define ACL_MAX_HW_NUM           128
#define ACL_MAX_SHAPE_COUNT      128
#define ACL_INVALID_NODE_INDEX   0xFFFFFFFF

#define ACL_MDL_LOAD_FROM_FILE            1
#define ACL_MDL_LOAD_FROM_FILE_WITH_MEM   2
#define ACL_MDL_LOAD_FROM_MEM             3
#define ACL_MDL_LOAD_FROM_MEM_WITH_MEM    4
#define ACL_MDL_LOAD_FROM_FILE_WITH_Q     5
#define ACL_MDL_LOAD_FROM_MEM_WITH_Q      6

#define ACL_DYNAMIC_TENSOR_NAME "ascend_mbatch_shape_data"
#define ACL_DYNAMIC_AIPP_NAME "ascend_dynamic_aipp_data"
#define ACL_ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES "_datadump_original_op_names"

/* used for ACL_MDL_WORKSPACE_MEM_OPTIMIZE */
#define ACL_WORKSPACE_MEM_OPTIMIZE_DEFAULT 0
#define ACL_WORKSPACE_MEM_OPTIMIZE_INPUTOUTPUT 1

typedef struct aclmdlDataset aclmdlDataset;
typedef struct aclmdlDesc aclmdlDesc;
typedef struct aclmdlAIPP aclmdlAIPP;
typedef struct aclAippExtendInfo aclAippExtendInfo;
typedef struct aclmdlConfigHandle aclmdlConfigHandle;
typedef struct aclmdlExecConfigHandle aclmdlExecConfigHandle;

typedef enum {
    ACL_YUV420SP_U8 = 1,
    ACL_XRGB8888_U8 = 2,
    ACL_RGB888_U8 = 3,
    ACL_YUV400_U8 = 4,
    ACL_NC1HWC0DI_FP16 = 5,
    ACL_NC1HWC0DI_S8 = 6,
    ACL_ARGB8888_U8 = 7,
    ACL_YUYV_U8 = 8,
    ACL_YUV422SP_U8 = 9,
    ACL_AYUV444_U8 = 10,
    ACL_RAW10 = 11,
    ACL_RAW12 = 12,
    ACL_RAW16 = 13,
    ACL_RAW24 = 14,
    ACL_AIPP_RESERVED = 0xFFFF,
} aclAippInputFormat;

typedef enum {
    ACL_MDL_PRIORITY_INT32 = 0,
    ACL_MDL_LOAD_TYPE_SIZET,
    ACL_MDL_PATH_PTR, /**< pointer to model load path with deep copy */
    ACL_MDL_MEM_ADDR_PTR, /**< pointer to model memory with shallow copy */
    ACL_MDL_MEM_SIZET,
    ACL_MDL_WEIGHT_ADDR_PTR, /**< pointer to weight memory of model with shallow copy */
    ACL_MDL_WEIGHT_SIZET,
    ACL_MDL_WORKSPACE_ADDR_PTR, /**< pointer to worksapce memory of model with shallow copy */
    ACL_MDL_WORKSPACE_SIZET,
    ACL_MDL_INPUTQ_NUM_SIZET,
    ACL_MDL_INPUTQ_ADDR_PTR, /**< pointer to inputQ with shallow copy */
    ACL_MDL_OUTPUTQ_NUM_SIZET,
    ACL_MDL_OUTPUTQ_ADDR_PTR, /**< pointer to outputQ with shallow copy */
    ACL_MDL_WORKSPACE_MEM_OPTIMIZE
} aclmdlConfigAttr;

typedef enum {
    ACL_MDL_STREAM_SYNC_TIMEOUT = 0,
    ACL_MDL_EVENT_SYNC_TIMEOUT,
    ACL_MDL_WORK_ADDR_PTR, /**< param */
    ACL_MDL_WORK_SIZET, /**< param */
    ACL_MDL_MPAIMID_SIZET, /**< param reserved */
    ACL_MDL_AICQOS_SIZET, /**< param reserved */
    ACL_MDL_AICOST_SIZET, /**< param reserved */
    ACL_MDL_MEC_TIMETHR_SIZET /**< param reserved */
} aclmdlExecConfigAttr;

typedef enum {
    ACL_DATA_WITHOUT_AIPP = 0,
    ACL_DATA_WITH_STATIC_AIPP,
    ACL_DATA_WITH_DYNAMIC_AIPP,
    ACL_DYNAMIC_AIPP_NODE
} aclmdlInputAippType;

typedef struct aclmdlIODims {
    char name[ACL_MAX_TENSOR_NAME_LEN]; /**< tensor name */
    size_t dimCount;  /**< dim array count */
    int64_t dims[ACL_MAX_DIM_CNT]; /**< dim data array */
} aclmdlIODims;

typedef struct aclAippDims {
    aclmdlIODims srcDims; /**< input dims before model transform */
    size_t srcSize; /**< input size before model transform */
    aclmdlIODims aippOutdims; /**< aipp output dims */
    size_t aippOutSize; /**< aipp output size */
} aclAippDims;

typedef struct aclmdlBatch {
    size_t batchCount; /**< batch array count */
    uint64_t batch[ACL_MAX_BATCH_NUM]; /**< batch data array */
} aclmdlBatch;

typedef struct aclmdlHW {
    size_t hwCount; /**< height&width array count */
    uint64_t hw[ACL_MAX_HW_NUM][2]; /**< height&width data array */
} aclmdlHW;

typedef struct aclAippInfo {
    aclAippInputFormat inputFormat;
    int32_t srcImageSizeW;
    int32_t srcImageSizeH;
    int8_t cropSwitch;
    int32_t loadStartPosW;
    int32_t loadStartPosH;
    int32_t cropSizeW;
    int32_t cropSizeH;
    int8_t resizeSwitch;
    int32_t resizeOutputW;
    int32_t resizeOutputH;
    int8_t paddingSwitch;
    int32_t leftPaddingSize;
    int32_t rightPaddingSize;
    int32_t topPaddingSize;
    int32_t bottomPaddingSize;
    int8_t cscSwitch;
    int8_t rbuvSwapSwitch;
    int8_t axSwapSwitch;
    int8_t singleLineMode;
    int32_t matrixR0C0;
    int32_t matrixR0C1;
    int32_t matrixR0C2;
    int32_t matrixR1C0;
    int32_t matrixR1C1;
    int32_t matrixR1C2;
    int32_t matrixR2C0;
    int32_t matrixR2C1;
    int32_t matrixR2C2;
    int32_t outputBias0;
    int32_t outputBias1;
    int32_t outputBias2;
    int32_t inputBias0;
    int32_t inputBias1;
    int32_t inputBias2;
    int32_t meanChn0;
    int32_t meanChn1;
    int32_t meanChn2;
    int32_t meanChn3;
    float minChn0;
    float minChn1;
    float minChn2;
    float minChn3;
    float varReciChn0;
    float varReciChn1;
    float varReciChn2;
    float varReciChn3;
    aclFormat srcFormat;
    aclDataType srcDatatype;
    size_t srcDimNum;
    size_t shapeCount;
    aclAippDims outDims[ACL_MAX_SHAPE_COUNT];
    aclAippExtendInfo *aippExtend; /**< reserved parameters, current version needs to be null */
} aclAippInfo;

/**
 * @ingroup AscendCL
 * @brief Create data of type aclmdlDesc
 *
 * @retval the aclmdlDesc pointer
 */
ACL_FUNC_VISIBILITY aclmdlDesc *aclmdlCreateDesc();

/**
 * @ingroup AscendCL
 * @brief destroy data of type aclmdlDesc
 *
 * @param modelDesc [IN]   Pointer to almdldlDesc to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc);

/**
 * @ingroup AscendCL
 * @brief Get aclmdlDesc data of the model according to the model ID
 *
 * @param  modelDesc [OUT]   aclmdlDesc pointer
 * @param  modelId [IN]      model id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId);

/**
 * @ingroup AscendCL
 * @brief Get aclmdlDesc data of the model according to the model path
 *
 * @param  modelDesc [OUT]   aclmdlDesc pointer
 * @param  modelPath [IN]    model path
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetDescFromFile(aclmdlDesc *modelDesc, const char *modelPath);

/**
 * @ingroup AscendCL
 * @brief Get aclmdlDesc data of the model according to the model and modelSize
 *
 * @param  modelDesc [OUT]   aclmdlDesc pointer
 * @param  model [IN]        model pointer
 * @param  modelSize [IN]    model size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetDescFromMem(aclmdlDesc *modelDesc, const void *model, size_t modelSize);

/**
 * @ingroup AscendCL
 * @brief Get the number of the inputs of
 *        the model according to data of aclmdlDesc
 *
 * @param  modelDesc [IN]   aclmdlDesc pointer
 *
 * @retval input size with aclmdlDesc
 */
ACL_FUNC_VISIBILITY size_t aclmdlGetNumInputs(aclmdlDesc *modelDesc);

/**
 * @ingroup AscendCL
 * @brief Get the number of the output of
 *        the model according to data of aclmdlDesc
 *
 * @param  modelDesc [IN]   aclmdlDesc pointer
 *
 * @retval output size with aclmdlDesc
 */
ACL_FUNC_VISIBILITY size_t aclmdlGetNumOutputs(aclmdlDesc *modelDesc);

/**
 * @ingroup AscendCL
 * @brief Get the size of the specified input according to
 *        the data of type aclmdlDesc
 *
 * @param  modelDesc [IN]  aclmdlDesc pointer
 * @param  index [IN] the size of the number of inputs to be obtained,
 *         the index value starts from 0
 *
 * @retval Specify the size of the input
 */
ACL_FUNC_VISIBILITY size_t aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief Get the size of the specified output according to
 *        the data of type aclmdlDesc
 *
 * @param modelDesc [IN]   aclmdlDesc pointer
 * @param index [IN]  the size of the number of outputs to be obtained,
 *        the index value starts from 0
 *
 * @retval Specify the size of the output
 */
ACL_FUNC_VISIBILITY size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief Create config handle of execute
 *
 * @retval the aclmdlCreateExecConfigHandle pointer
 */
ACL_FUNC_VISIBILITY aclmdlExecConfigHandle *aclmdlCreateExecConfigHandle();

/**
 * @ingroup AscendCL
 * @brief Destroy config handle of model execute
 *
 * @param  handle [IN]  Pointer to aclmdlExecConfigHandle to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlDestroyExecConfigHandle(const aclmdlExecConfigHandle *handle);

/**
 * @ingroup AscendCL
 * @brief Create data of type aclmdlDataset
 *
 * @retval the aclmdlDataset pointer
 */
ACL_FUNC_VISIBILITY aclmdlDataset *aclmdlCreateDataset();

/**
 * @ingroup AscendCL
 * @brief destroy data of type aclmdlDataset
 *
 * @param  dataset [IN]  Pointer to aclmdlDataset to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlDestroyDataset(const aclmdlDataset *dataset);

/**
 * @ingroup AscendCL
 * @brief Add aclDataBuffer to aclmdlDataset
 *
 * @param dataset [OUT]    aclmdlDataset address of aclDataBuffer to be added
 * @param dataBuffer [IN]  aclDataBuffer address to be added
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief Set aclTensorDesc to aclmdlDataset
 *
 * @param dataset [OUT]    aclmdlDataset address of aclDataBuffer to be added
 * @param tensorDesc [IN]  aclTensorDesc address to be added
 * @param index [IN]       index of tensorDesc which to be added
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetDatasetTensorDesc(aclmdlDataset *dataset,
                                                        aclTensorDesc *tensorDesc,
                                                        size_t index);

/**
 * @ingroup AscendCL
 * @brief Get aclTensorDesc from aclmdlDataset
 *
 * @param dataset [IN]    aclmdlDataset pointer;
 * @param index [IN]      index of tensorDesc
 *
 * @retval Get address of aclTensorDesc when executed successfully.
 * @retval Failure return NULL
 */
ACL_FUNC_VISIBILITY aclTensorDesc *aclmdlGetDatasetTensorDesc(const aclmdlDataset *dataset, size_t index);

/**
 * @ingroup AscendCL
 * @brief Get the number of aclDataBuffer in aclmdlDataset
 *
 * @param dataset [IN]   aclmdlDataset pointer
 *
 * @retval the number of aclDataBuffer
 */
ACL_FUNC_VISIBILITY size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataset);

/**
 * @ingroup AscendCL
 * @brief Get the aclDataBuffer in aclmdlDataset by index
 *
 * @param dataset [IN]   aclmdlDataset pointer
 * @param index [IN]     the index of aclDataBuffer
 *
 * @retval Get successfully, return the address of aclDataBuffer
 * @retval Failure return NULL
 */
ACL_FUNC_VISIBILITY aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index);

/**
 * @ingroup AscendCL
 * @brief Load offline model data from files
 * and manage memory internally by the system
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations
 *
 * @param modelPath [IN]   Storage path for offline model files
 * @param modelId [OUT]    Model ID generated after
 *        the system finishes loading the model
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromFile(const char *modelPath, uint32_t *modelId);

/**
 * @ingroup AscendCL
 * @brief Load offline model data from memory and manage the memory of
 * model running internally by the system
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations
 *
 * @param model [IN]      Model data stored in memory
 * @param modelSize [IN]  model data size
 * @param modelId [OUT]   Model ID generated after
 *        the system finishes loading the model
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromMem(const void *model,  size_t modelSize,
                                               uint32_t *modelId);

/**
 * @ingroup AscendCL
 * @brief Load offline model data from a file,
 * and the user manages the memory of the model run by itself
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations.
 * @param modelPath [IN]   Storage path for offline model files
 * @param modelId [OUT]    Model ID generated after finishes loading the model
 * @param workPtr [IN]     A pointer to the working memory
 *                         required by the model on the Device,can be null
 * @param workSize [IN]    The amount of working memory required by the model
 * @param weightPtr [IN]   Pointer to model weight memory on Device
 * @param weightSize [IN]  The amount of weight memory required by the model
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromFileWithMem(const char *modelPath,
                                                       uint32_t *modelId, void *workPtr, size_t workSize,
                                                       void *weightPtr, size_t weightSize);

/**
 * @ingroup AscendCL
 * @brief Load offline model data from memory,
 * and the user can manage the memory of model running
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations
 * @param model [IN]      Model data stored in memory
 * @param modelSize [IN]  model data size
 * @param modelId [OUT]   Model ID generated after finishes loading the model
 * @param workPtr [IN]    A pointer to the working memory
 *                        required by the model on the Device,can be null
 * @param workSize [IN]   work memory size
 * @param weightPtr [IN]  Pointer to model weight memory on Device,can be null
 * @param weightSize [IN] The amount of weight memory required by the model
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromMemWithMem(const void *model, size_t modelSize,
                                                      uint32_t *modelId, void *workPtr, size_t workSize,
                                                      void *weightPtr, size_t weightSize);

/**
 * @ingroup AscendCL
 * @brief load model from file with async queue
 *
 * @param modelPath  [IN] model path
 * @param modelId [OUT]   return model id if load success
 * @param inputQ [IN]     input queue pointer
 * @param inputQNum [IN]  input queue num
 * @param outputQ [IN]    output queue pointer
 * @param outputQNum [IN] output queue num
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromFileWithQ(const char *modelPath, uint32_t *modelId, const uint32_t *inputQ,
                                                     size_t inputQNum, const uint32_t *outputQ, size_t outputQNum);

/**
 * @ingroup AscendCL
 * @brief load model from memory with async queue
 *
 * @param model [IN]      model memory which user manages
 * @param modelSize [IN]  model size
 * @param modelId [OUT]   return model id if load success
 * @param inputQ [IN]     input queue pointer
 * @param inputQNum [IN]  input queue num
 * @param outputQ [IN]    output queue pointer
 * @param outputQNum [IN] output queue num
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromMemWithQ(const void *model, size_t modelSize, uint32_t *modelId,
                                                    const uint32_t *inputQ, size_t inputQNum,
                                                    const uint32_t *outputQ, size_t outputQNum);

/**
 * @ingroup AscendCL
 * @brief Execute model synchronous inference until the inference result is returned
 *
 * @param  modelId [IN]   ID of the model to perform inference
 * @param  input [IN]     Input data for model inference
 * @param  output [OUT]   Output data for model inference
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output);

/**
 * @ingroup AscendCL
 * @brief Execute model synchronous inference until the inference result is returned
 *
 * @param  modelId [IN]   ID of the model to perform inference
 * @param  input [IN]     Input data for model inference
 * @param  output [OUT]   Output data for model inference
 * @param  stream [IN]   stream
 * @param  handle [IN]   config of model execute
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlExecuteV2(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output,
                                             aclrtStream stream, const aclmdlExecConfigHandle *handle);

/**
 * @ingroup AscendCL
 * @brief Execute model asynchronous inference until the inference result is returned
 *
 * @param  modelId [IN]   ID of the model to perform inference
 * @param  input [IN]     Input data for model inference
 * @param  output [OUT]   Output data for model inference
 * @param  stream [IN]   stream
 * @param  handle [IN]   config of model execute
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY  aclError aclmdlExecuteAsyncV2(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output,
                                                   aclrtStream stream, const aclmdlExecConfigHandle *handle);
/**
 * @ingroup AscendCL
 * @brief Execute model asynchronous inference until the inference result is returned
 *
 * @param  modelId [IN]   ID of the model to perform inference
 * @param  input [IN]     Input data for model inference
 * @param  output [OUT]   Output data for model inference
 * @param  stream [IN]    stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem
 */
ACL_FUNC_VISIBILITY aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *input,
                                                aclmdlDataset *output, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief unload model with model id
 *
 * @param  modelId [IN]   model id to be unloaded
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlUnload(uint32_t modelId);

/**
 * @ingroup AscendCL
 * @brief Get the weight memory size and working memory size
 * required for model execution according to the model file
 *
 * @param  fileName [IN]     Model path to get memory information
 * @param  workSize [OUT]    The amount of working memory for model executed
 * @param  weightSize [OUT]  The amount of weight memory for model executed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlQuerySize(const char *fileName, size_t *workSize, size_t *weightSize);

/**
 * @ingroup AscendCL
 * @brief Obtain the weights required for
 * model execution according to the model data in memory
 *
 * @par Restriction
 * The execution and weight memory is Device memory,
 * and requires user application and release.
 * @param  model [IN]        model memory which user manages
 * @param  modelSize [IN]    model data size
 * @param  workSize [OUT]    The amount of working memory for model executed
 * @param  weightSize [OUT]  The amount of weight memory for model executed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlQuerySizeFromMem(const void *model, size_t modelSize, size_t *workSize,
                                                    size_t *weightSize);

/**
 * @ingroup AscendCL
 * @brief In dynamic batch scenarios,
 * it is used to set the number of images processed
 * at one time during model inference
 *
 * @param  modelId [IN]     model id
 * @param  dataset [IN|OUT] data for model inference
 * @param  index [IN]       index of dynamic tensor
 * @param  batchSize [IN]   Number of images processed at a time during model
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetDynamicBatchSize(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                       uint64_t batchSize);

/**
 * @ingroup AscendCL
 * @brief Sets the H and W of the specified input of the model
 *
 * @param  modelId [IN]     model id
 * @param  dataset [IN|OUT] data for model inference
 * @param  index [IN]       index of dynamic tensor
 * @param  height [IN]      model height
 * @param  width [IN]       model width
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetDynamicHWSize(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                    uint64_t height, uint64_t width);

/**
 * @ingroup AscendCL
 * @brief Sets the dynamic dims of the specified input of the model
 *
 * @param  modelId [IN]     model id
 * @param  dataset [IN|OUT] data for model inference
 * @param  index [IN]       index of dynamic dims
 * @param  dims [IN]        value of dynamic dims
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetInputDynamicDims(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                       const aclmdlIODims *dims);

/**
 * @ingroup AscendCL
 * @brief get input dims info
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  input tensor index
 * @param dims [OUT]  dims info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlGetInputDimsV2
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

/**
 * @ingroup AscendCL
 * @brief get input dims info(version 2), especially for static aipp
 * it is the same with aclmdlGetInputDims while model without static aipp
 *
 * @param modelDesc [IN] model description
 * @param index [IN]     input tensor index
 * @param dims [OUT]     dims info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlGetInputDims
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetInputDimsV2(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

/**
 * @ingroup AscendCL
 * @brief get output dims info
 *
 * @param modelDesc [IN] model description
 * @param index [IN]     output tensor index
 * @param dims [OUT]     dims info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

/**
 * @ingroup AscendCL
 * @brief get current output dims info
 *
 * @par Function
 * The following use cases are supported:
 * @li Get current output shape when model is dynamic and
 * dynamic shape info is set
 * @li Get max output shape when model is dynamic and
 * dynamic shape info is not set
 * @li Get actual output shape when model is static
 *
 * @param modelDesc [IN] model description
 * @param index [IN]     output tensor index
 * @param dims [OUT]     dims info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetCurOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

/**
 * @ingroup AscendCL
 * @brief get attr value by op name
 *
 * @param modelDesc [IN]   model description
 * @param opName [IN]      op name
 * @param attr [IN]        attr name
 *
 * @retval the attr value
 */
ACL_FUNC_VISIBILITY const char *aclmdlGetOpAttr(aclmdlDesc *modelDesc, const char *opName, const char *attr);

/**
 * @ingroup AscendCL
 * @brief get input name by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      intput tensor index
 *
 * @retval input tensor name,the same life cycle with modelDesc
 */
ACL_FUNC_VISIBILITY const char *aclmdlGetInputNameByIndex(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get output name by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      output tensor index
 *
 * @retval output tensor name,the same life cycle with modelDesc
 */
ACL_FUNC_VISIBILITY const char *aclmdlGetOutputNameByIndex(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get input format by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      intput tensor index
 *
 * @retval input tensor format
 */
ACL_FUNC_VISIBILITY aclFormat aclmdlGetInputFormat(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get output format by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      output tensor index
 *
 * @retval output tensor format
 */
ACL_FUNC_VISIBILITY aclFormat aclmdlGetOutputFormat(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get input data type by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  intput tensor index
 *
 * @retval input tensor data type
 */
ACL_FUNC_VISIBILITY aclDataType aclmdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get output data type by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  output tensor index
 *
 * @retval output tensor data type
 */
ACL_FUNC_VISIBILITY aclDataType aclmdlGetOutputDataType(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get input tensor index by name
 *
 * @param modelDesc [IN]  model description
 * @param name [IN]    intput tensor name
 * @param index [OUT]  intput tensor index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetInputIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *index);

/**
 * @ingroup AscendCL
 * @brief get output tensor index by name
 *
 * @param modelDesc [IN]  model description
 * @param name [IN]  output tensor name
 * @param index [OUT]  output tensor index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetOutputIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *index);

/**
 * @ingroup AscendCL
 * @brief get dynamic batch info
 *
 * @param modelDesc [IN]  model description
 * @param batch [OUT]  dynamic batch info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetDynamicBatch(const aclmdlDesc *modelDesc, aclmdlBatch *batch);

/**
 * @ingroup AscendCL
 * @brief get dynamic height&width info
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  input tensor index
 * @param hw [OUT]  dynamic height&width info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetDynamicHW(const aclmdlDesc *modelDesc, size_t index, aclmdlHW *hw);

/**
 * @ingroup AscendCL
 * @brief get dynamic gear count
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  unused, must be -1
 * @param gearCount [OUT]  dynamic gear count
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetInputDynamicGearCount(const aclmdlDesc *modelDesc, size_t index,
                                                            size_t *gearCount);

/**
 * @ingroup AscendCL
 * @brief get dynamic dims info
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  unused, must be -1
 * @param dims [OUT]  value of dynamic dims
 * @param gearCount [IN]  dynamic gear count
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetInputDynamicDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims,
                                                       size_t gearCount);

/**
 * @ingroup AscendCL
 * @brief Create data of type aclmdlAIPP
 *
 * @param batchSize [IN]    batchsizes of model
 *
 * @retval the aclmdlAIPP pointer
 */
ACL_FUNC_VISIBILITY aclmdlAIPP *aclmdlCreateAIPP(uint64_t batchSize);

/**
 * @ingroup AscendCL
 * @brief destroy data of type aclmdlAIPP
 *
 * @param aippParmsSet [IN]    Pointer for aclmdlAIPP to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlDestroyAIPP(const aclmdlAIPP *aippParmsSet);

/**
 * @ingroup AscendCL
 * @brief set InputFormat of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param inputFormat [IN]    The inputFormat of aipp
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPInputFormat(aclmdlAIPP *aippParmsSet, aclAippInputFormat inputFormat);

/**
 * @ingroup AscendCL
 * @brief set cscParms of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]    Pointer for aclmdlAIPP
 * @param csc_switch [IN]       Csc switch
 * @param cscMatrixR0C0 [IN]    Csc_matrix_r0_c0
 * @param cscMatrixR0C1 [IN]    Csc_matrix_r0_c1
 * @param cscMatrixR0C2 [IN]    Csc_matrix_r0_c2
 * @param cscMatrixR1C0 [IN]    Csc_matrix_r1_c0
 * @param cscMatrixR1C1 [IN]    Csc_matrix_r1_c1
 * @param cscMatrixR1C2 [IN]    Csc_matrix_r1_c2
 * @param cscMatrixR2C0 [IN]    Csc_matrix_r2_c0
 * @param cscMatrixR2C1 [IN]    Csc_matrix_r2_c1
 * @param cscMatrixR2C2 [IN]    Csc_matrix_r2_c2
 * @param cscOutputBiasR0 [IN]  Output Bias for RGB to YUV, element of row 0, unsigned number
 * @param cscOutputBiasR1 [IN]  Output Bias for RGB to YUV, element of row 1, unsigned number
 * @param cscOutputBiasR2 [IN]  Output Bias for RGB to YUV, element of row 2, unsigned number
 * @param cscInputBiasR0 [IN]   Input Bias for YUV to RGB, element of row 0, unsigned number
 * @param cscInputBiasR1 [IN]   Input Bias for YUV to RGB, element of row 1, unsigned number
 * @param cscInputBiasR2 [IN]   Input Bias for YUV to RGB, element of row 2, unsigned number
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPCscParams(aclmdlAIPP *aippParmsSet, int8_t cscSwitch,
                                                    int16_t cscMatrixR0C0, int16_t cscMatrixR0C1, int16_t cscMatrixR0C2,
                                                    int16_t cscMatrixR1C0, int16_t cscMatrixR1C1, int16_t cscMatrixR1C2,
                                                    int16_t cscMatrixR2C0, int16_t cscMatrixR2C1, int16_t cscMatrixR2C2,
                                                    uint8_t cscOutputBiasR0, uint8_t cscOutputBiasR1,
                                                    uint8_t cscOutputBiasR2, uint8_t cscInputBiasR0,
                                                    uint8_t cscInputBiasR1, uint8_t cscInputBiasR2);

/**
 * @ingroup AscendCL
 * @brief set rb/ub swap switch of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param rbuvSwapSwitch [IN] rb/ub swap switch
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPRbuvSwapSwitch(aclmdlAIPP *aippParmsSet, int8_t rbuvSwapSwitch);

/**
 * @ingroup AscendCL
 * @brief set RGBA->ARGB, YUVA->AYUV swap switch of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param axSwapSwitch [IN]   RGBA->ARGB, YUVA->AYUV swap switch
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPAxSwapSwitch(aclmdlAIPP *aippParmsSet, int8_t axSwapSwitch);

/**
 * @ingroup AscendCL
 * @brief set source image of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param srcImageSizeW [IN]  Source image width
 * @param srcImageSizeH [IN]  Source image height
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPSrcImageSize(aclmdlAIPP *aippParmsSet, int32_t srcImageSizeW,
                                                       int32_t srcImageSizeH);

/**
 * @ingroup AscendCL
 * @brief set resize switch of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param scfSwitch [IN]      Resize switch
 * @param scfInputSizeW [IN]  Input width of scf
 * @param scfInputSizeH [IN]  Input height of scf
 * @param scfOutputSizeW [IN] Output width of scf
 * @param scfOutputSizeH [IN] Output height of scf
 * @param batchIndex [IN]     Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPScfParams(aclmdlAIPP *aippParmsSet,
                                                    int8_t scfSwitch,
                                                    int32_t scfInputSizeW,
                                                    int32_t scfInputSizeH,
                                                    int32_t scfOutputSizeW,
                                                    int32_t scfOutputSizeH,
                                                    uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set cropParams of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param cropSwitch [IN]     Crop switch
 * @param cropStartPosW [IN]  The start horizontal position of cropping
 * @param cropStartPosH [IN]  The start vertical position of cropping
 * @param cropSizeW [IN]      Crop width
 * @param cropSizeH [IN]      Crop height
 * @param batchIndex [IN]     Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPCropParams(aclmdlAIPP *aippParmsSet,
                                                     int8_t cropSwitch,
                                                     int32_t cropStartPosW,
                                                     int32_t cropStartPosH,
                                                     int32_t cropSizeW,
                                                     int32_t cropSizeH,
                                                     uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set paddingParams of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]      Pointer for aclmdlAIPP
 * @param paddingSwitch [IN]      Padding switch
 * @param paddingSizeTop [IN]     Top padding size
 * @param paddingSizeBottom [IN]  Bottom padding size
 * @param paddingSizeLeft [IN]    Left padding size
 * @param paddingSizeRight [IN]   Right padding size
 * @param batchIndex [IN]         Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPPaddingParams(aclmdlAIPP *aippParmsSet, int8_t paddingSwitch,
                                                        int32_t paddingSizeTop, int32_t paddingSizeBottom,
                                                        int32_t paddingSizeLeft, int32_t paddingSizeRight,
                                                        uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set DtcPixelMean of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]      Pointer for aclmdlAIPP
 * @param dtcPixelMeanChn0 [IN]   Mean value of channel 0
 * @param dtcPixelMeanChn1 [IN]   Mean value of channel 1
 * @param dtcPixelMeanChn2 [IN]   Mean value of channel 2
 * @param dtcPixelMeanChn3 [IN]   Mean value of channel 3
 * @param batchIndex [IN]         Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPDtcPixelMean(aclmdlAIPP *aippParmsSet,
                                                       int16_t dtcPixelMeanChn0,
                                                       int16_t dtcPixelMeanChn1,
                                                       int16_t dtcPixelMeanChn2,
                                                       int16_t dtcPixelMeanChn3,
                                                       uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set DtcPixelMin of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]    Pointer for aclmdlAIPP
 * @param dtcPixelMinChn0 [IN]  Min value of channel 0
 * @param dtcPixelMinChn1 [IN]  Min value of channel 1
 * @param dtcPixelMinChn2 [IN]  Min value of channel 2
 * @param dtcPixelMinChn3 [IN]  Min value of channel 3
 * @param batchIndex [IN]       Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPDtcPixelMin(aclmdlAIPP *aippParmsSet,
                                                      float dtcPixelMinChn0,
                                                      float dtcPixelMinChn1,
                                                      float dtcPixelMinChn2,
                                                      float dtcPixelMinChn3,
                                                      uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set PixelVarReci of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]       Pointer for aclmdlAIPP
 * @param dtcPixelVarReciChn0 [IN] sfr_dtc_pixel_variance_reci_ch0
 * @param dtcPixelVarReciChn1 [IN] sfr_dtc_pixel_variance_reci_ch1
 * @param dtcPixelVarReciChn2 [IN] sfr_dtc_pixel_variance_reci_ch2
 * @param dtcPixelVarReciChn3 [IN] sfr_dtc_pixel_variance_reci_ch3
 * @param batchIndex [IN]          Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPPixelVarReci(aclmdlAIPP *aippParmsSet,
                                                       float dtcPixelVarReciChn0,
                                                       float dtcPixelVarReciChn1,
                                                       float dtcPixelVarReciChn2,
                                                       float dtcPixelVarReciChn3,
                                                       uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set aipp parameters to model
 *
 * @param modelId [IN]        model id
 * @param dataset [IN]        Pointer of dataset
 * @param index [IN]          index of input for aipp data(ACL_DYNAMIC_AIPP_NODE)
 * @param aippParmsSet [IN]   Pointer for aclmdlAIPP
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName | aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetInputAIPP(uint32_t modelId,
                                                aclmdlDataset *dataset,
                                                size_t index,
                                                const aclmdlAIPP *aippParmsSet);

/**
 * @ingroup AscendCL
 * @brief set aipp parameters to model
 *
 * @param modelId [IN]        model id
 * @param dataset [IN]        Pointer of dataset
 * @param index [IN]          index of input for data which linked dynamic aipp(ACL_DATA_WITH_DYNAMIC_AIPP)
 * @param aippParmsSet [IN]   Pointer for aclmdlAIPP
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName | aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPByInputIndex(uint32_t modelId,
                                                       aclmdlDataset *dataset,
                                                       size_t index,
                                                       const aclmdlAIPP *aippParmsSet);

/**
 * @ingroup AscendCL
 * @brief get input aipp type
 *
 * @param modelId [IN]        model id
 * @param index [IN]          index of input
 * @param type [OUT]          aipp type for input.refrer to aclmdlInputAippType(enum)
 * @param dynamicAttachedDataIndex [OUT]     index for dynamic attached data(ACL_DYNAMIC_AIPP_NODE)
 *        valid when type is ACL_DATA_WITH_DYNAMIC_AIPP, invalid value is ACL_INVALID_NODE_INDEX
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName | aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlGetAippType(uint32_t modelId,
                                               size_t index,
                                               aclmdlInputAippType *type,
                                               size_t *dynamicAttachedDataIndex);

/**
 * @ingroup AscendCL
 * @brief get static aipp parameters from model
 *
 * @param modelId [IN]        model id
 * @param index [IN]          index of tensor
 * @param aippInfo [OUT]      Pointer for static aipp info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_MODEL_AIPP_NOT_EXIST The tensor of index is not configured with aipp
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName
*/
ACL_FUNC_VISIBILITY aclError aclmdlGetFirstAippInfo(uint32_t modelId, size_t index, aclAippInfo *aippInfo);

/**
 * @ingroup AscendCL
 * @brief get op description info
 *
 * @param deviceId [IN]       device id
 * @param streamId [IN]       stream id
 * @param taskId [IN]         task id
 * @param opName [OUT]        pointer to op name
 * @param opNameLen [IN]      the length of op name
 * @param inputDesc [OUT]     pointer to input description
 * @param numInputs [OUT]     the number of input tensor
 * @param outputDesc [OUT]    pointer to output description
 * @param numOutputs [OUT]    the number of output tensor
 *
 * @retval ACL_SUCCESS The function is successfully executed
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlCreateAndGetOpDesc(uint32_t deviceId, uint32_t streamId,
    uint32_t taskId, char *opName, size_t opNameLen, aclTensorDesc **inputDesc, size_t *numInputs,
    aclTensorDesc **outputDesc, size_t *numOutputs);

/**
 * @ingroup AscendCL
 * @brief init dump
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlInitDump();

/**
 * @ingroup AscendCL
 * @brief set param of dump
 *
 * @param dumpCfgPath [IN]   the path of dump config
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetDump(const char *dumpCfgPath);

/**
 * @ingroup AscendCL
 * @brief finalize dump.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlFinalizeDump();

/**
 * @ingroup AscendCL
 * @brief load model with config
 *
 * @param handle [IN]    pointer to model config handle
 * @param modelId [OUT]  pointer to model id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlLoadWithConfig(const aclmdlConfigHandle *handle, uint32_t *modelId);

/**
 * @ingroup AscendCL
 * @brief create model config handle of type aclmdlConfigHandle
 *
 * @retval the aclmdlConfigHandle pointer
 *
 * @see aclmdlDestroyConfigHandle
*/
ACL_FUNC_VISIBILITY aclmdlConfigHandle *aclmdlCreateConfigHandle();

/**
 * @ingroup AscendCL
 * @brief destroy data of type aclmdlConfigHandle
 *
 * @param handle [IN]   pointer to model config handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateConfigHandle
 */
ACL_FUNC_VISIBILITY aclError aclmdlDestroyConfigHandle(aclmdlConfigHandle *handle);

/**
 * @ingroup AscendCL
 * @brief set config for model load
 *
 * @param handle [OUT]    pointer to model config handle
 * @param attr [IN]       config attr in model config handle to be set
 * @param attrValue [IN]  pointer to model config value
 * @param valueSize [IN]  memory size of attrValue
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetConfigOpt(aclmdlConfigHandle *handle, aclmdlConfigAttr attr,
    const void *attrValue, size_t valueSize);

/**
 * @ingroup AscendCL
 * @brief set config for model execute
 *
 * @param handle [OUT]    pointer to model execute config handle
 * @param attr [IN]       config attr in model execute config handle to be set
 * @param attrValue [IN]  pointer to model execute config value
 * @param valueSize [IN]  memory size of attrValue
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetExecConfigOpt(aclmdlExecConfigHandle *handle, aclmdlExecConfigAttr attr,
                                                    const void *attrValue, size_t valueSize);

/**
 * @ingroup AscendCL
 * @brief get real tensor name from modelDesc
 *
 * @param modelDesc [IN]  pointer to modelDesc
 * @param name [IN]       tensor name
 *
 * @retval the pointer of real tensor name
 * @retval Failure return NULL
 */
ACL_FUNC_VISIBILITY const char *aclmdlGetTensorRealName(const aclmdlDesc *modelDesc, const char *name);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_MODEL_H_


/**
* @file acl.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_H_
#define INC_EXTERNAL_ACL_ACL_H_

// #include "acl_rt.h"
// #include "acl_op.h"
// #include "acl_mdl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Current version is 1.7.0
#define ACL_MAJOR_VERSION    1
#define ACL_MINOR_VERSION    7
#define ACL_PATCH_VERSION    0

/**
 * @ingroup AscendCL
 * @brief acl initialize
 *
 * @par Restriction
 * The aclInit interface can be called only once in a process
 * @param configPath [IN]    the config path,it can be NULL
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclInit(const char *configPath);

/**
 * @ingroup AscendCL
 * @brief acl finalize
 *
 * @par Restriction
 * Need to call aclFinalize before the process exits.
 * After calling aclFinalize,the services cannot continue to be used normally.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclFinalize();

/**
 * @ingroup AscendCL
 * @brief query ACL interface version
 *
 * @param majorVersion[OUT] ACL interface major version
 * @param minorVersion[OUT] ACL interface minor version
 * @param patchVersion[OUT] ACL interface patch version
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetVersion(int32_t *majorVersion, int32_t *minorVersion, int32_t *patchVersion);

/**
 * @ingroup AscendCL
 * @brief get recent error message
 *
 * @retval null for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY const char *aclGetRecentErrMsg();

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_H_
