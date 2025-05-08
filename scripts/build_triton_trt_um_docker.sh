FILE_DIR=$(dirname "$0")
PROJECT_DIR=$(cd "$FILE_DIR/.." && pwd)
DOCKER_FILE_PATH="$PROJECT_DIR/environment/Dockerfile.triton"


mkdir -p triton-docker-build
WORKSPACE_DIR=triton-docker-build

echo "Project directory: $PROJECT_DIR"
echo "Dockerfile: $DOCKER_FILE_PATH"
echo "Docker build directory: $WORKSPACE_DIR"
echo "Proxy: $https_proxy"
# Ask for confirmation before proceeding
read -p "Do you want to continue? (y/n): " confirm
if [[ $confirm != [yY] ]]; then
    echo "Build process aborted."
    exit 1
fi

# clone triton tensorrt um
git clone --recurse-submodules \
    git@github.com:SiriusInfTra/triton_tensorrt_um.git \
    "$WORKSPACE_DIR/triton_tensorrt_um"

cp ${PROJECT_DIR}/scripts/build_triton_trt_um.sh ${WORKSPACE_DIR} 

PROXY_ARG=""
if [ ! -z "$https_proxy" ]; then
  PROXY_ARG="--build-arg PROXY=$https_proxy"
  echo "Using proxy: $https_proxy"
fi

docker build \
    $PROXY_ARG \
    -t siriusinftra/triton-trt-um:latest \
    -f "$DOCKER_FILE_PATH" \
    "$WORKSPACE_DIR"