FILE_DIR=$(dirname "$0")
PROJECT_DIR=$(cd "$FILE_DIR/.." && pwd)
DOCKER_FILE_PATH="$PROJECT_DIR/environment/Dockerfile"

mkdir -p inftra-docker-build
WORKSPACE_DIR=$(pwd)/inftra-docker-build

echo "Project directory: $PROJECT_DIR"
echo "Dockerfile: $DOCKER_FILE_PATH"
echo "Docker build directory: $WORKSPACE_DIR"


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
  if [ ! -d "$repo_name" ]; then
    # git clone --recurse-submodules "$repo
    echo $repo
  else
    echo "Repository $repo_name already exists. Skipping clone."
  fi
done


docker build \
  --build-arg WORKSPACE_DIR="$WORKSPACE_DIR" \
  --build-arg PROJECT_DIR="$PROJECT_DIR" \
  -t inf-tra:latest \
  -f "$DOCKER_FILE_PATH" \
  "$WORKSPACE_DIR"