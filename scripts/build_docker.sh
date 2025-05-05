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
  git@ipads.se.sjtu.edu.cn:infer-train/pytorch.git 
  git@ipads.se.sjtu.edu.cn:infer-train/torch-vision.git 

  # TVM
  git@ipads.se.sjtu.edu.cn:infer-train/tvm.git

  # vLLM
  git@ipads.se.sjtu.edu.cn:infer-train/vllm.git
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
rsync -a --exclude="build" --exclude="build_Release" --exclude="build_Debug" \
  "$PROJECT_DIR"/ "$WORKSPACE_DIR"/gpu-col/

# Check if https_proxy environment variable exists and add it as build arg if it does
PROXY_ARG=""
if [ ! -z "$https_proxy" ]; then
  PROXY_ARG="--build-arg PROXY=$https_proxy"
  echo "Using proxy: $https_proxy"
fi

docker build \
  --build-arg PROJECT_DIR="gpu-col" \
  --build-arg DOCKER_BUILD_DIR="$WORKSPACE_DIR" \
  $PROXY_ARG \
  -t inf-tra:latest \
  -f "$DOCKER_FILE_PATH" \
  "$WORKSPACE_DIR"

