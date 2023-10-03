#ifndef COLSERVE_DLPACK_CONVERT_H
#define COLSERVE_DLPACK_CONVERT_H

#include <c10/core/TensorOptions.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <dlpack/dlpack.h>

namespace torch_col {

DLDataType getDLDataType(const at::ScalarType& scalar_type);
caffe2::TypeMeta getCaffeTypeMeta(const DLDataType& dtype);

}

#endif 