TORCH_HOME=$1
TORCH_VISION_HOME=$2

export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

cd $TORCH_HOME
python setup.py develop


cd $TORCH_VISION_HOME
python setup.py install