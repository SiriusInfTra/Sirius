FILE_DIR=$(dirname "$0")
PROJECT_DIR=$(cd "$FILE_DIR/.." && pwd)
DOCKER_FILE_PATH="$PROJECT_DIR/environment/Dockerfile"

mkdir -p inftra-docker-build
WORKSPACE_DIR=inftra-docker-build

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

# =========================================================
# clone and build dependencies
# =========================================================

repos=(
  # PyTorch related repos
  git@github.com:SiriusInfTra/pytorch.git 
  git@github.com:SiriusInfTra/torch-vision.git

  # TVM
  git@github.com:SiriusInfTra/tvm.git

  # vLLM
  git@github.com:SiriusInfTra/vllm.git
  git@github.com:SiriusInfTra/xformer.git
)

for repo in "${repos[@]}"; do
  repo_name=$(basename "$repo" .git)
  if [ ! -d "$WORKSPACE_DIR/$repo_name" ]; then
    git clone --recurse-submodules $repo "$WORKSPACE_DIR/$repo_name"
    echo $repo
  else
    echo "Repository $repo_name already exists. Skipping clone."
  fi
done

# TODO: ensure TVM/Triton models are copied to the workspace directory
rsync -a --exclude="$PROJECT_DIR/build" --exclude="build_Release" \
  --exclude="build_Debug" \
  --exclude="gpu-col-docker-log" --exclude="triton-model-wksp" \
  --exclude="mps-pipe-directory" \
  "$PROJECT_DIR"/ "$WORKSPACE_DIR"/gpu-col/


if [ ! -d "$WORKSPACE_DIR/tvm-models" ]; then
  mkdir -p "$WORKSPACE_DIR/tvm-models"
  touch "$WORKSPACE_DIR/tvm-models/keep"
fi

if [ ! -d "$WORKSPACE_DIR/triton-models" ]; then
  mkdir -p "$WORKSPACE_DIR/triton-models"
  touch "$WORKSPACE_DIR/triton-models/keep"
fi

# Check if https_proxy environment variable exists and add it as build arg if it does
PROXY_ARG=""
if [ ! -z "$https_proxy" ]; then
  PROXY_ARG="--build-arg PROXY=$https_proxy"
  echo "Using proxy: $https_proxy"
fi

DOCKER_BUILDKIT=1 docker build \
  --build-arg PROJECT_DIR="gpu-col" \
  --build-arg DOCKER_BUILD_DIR="$WORKSPACE_DIR" \
  $PROXY_ARG \
  -t siriusinftra/sirius:latest \
  -f "$DOCKER_FILE_PATH" \
  "$WORKSPACE_DIR"

