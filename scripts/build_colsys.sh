COLSYS_HOME=$1
TVM_HOME=$2
TORCH_HOME=$3
BOOST_HOME=$4

if [ -z "$CONDA_PREFIX" ]; then
  echo "Error: CONDA_PREFIX is not set. Please activate your conda environment."
  exit 1
fi

echo "COLSYS_HOME: $COLSYS_HOME | TVM_HOME: $TVM_HOME | TORCH_HOME: $TORCH_HOME | BOOST_HOME: $BOOST_HOME"

PYTHON_SITE_PACKAGES=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

cmake_cfg_cmd="cmake -DCMAKE_BUILD_TYPE:STRING=Release"
cmake_cfg_cmd+=" -DgRPC_INSTALL:STRING=ON -DgRPC_BUILD_TESTS:STRING=OFF"
cmake_cfg_cmd+=" -DCONDA_PREFIX:STRING=$CONDA_PREFIX"
cmake_cfg_cmd+=" \"-DCMAKE_PREFIX_PATH:STRING=$CONDA_PREFIX;$TORCH_HOME/torch/share/cmake;$PYTHON_SITE_PACKAGES/pybind11/share/cmake/\""
cmake_cfg_cmd+=" -DBoost_ROOT:STRING=$BOOST_HOME/install/lib/cmake/"
cmake_cfg_cmd+=" \"-DCMAKE_CUDA_ARCHITECTURES:STRING=70;80\""
cmake_cfg_cmd+=" -DTVM_HOME:STRING=$TVM_HOME"
cmake_cfg_cmd+=" -GNinja -S $COLSYS_HOME -B $COLSYS_HOME/build_Release"

echo "Running: $cmake_cfg_cmd"

eval $cmake_cfg_cmd

# build_Release will be linked to build
cd $COLSYS_HOME
cmake --build build