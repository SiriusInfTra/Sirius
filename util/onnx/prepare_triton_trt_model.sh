if [ -z "$SIRIUS_TENSORRT_HOME" ]; then
  echo "Please set the environment variable SIRIUS_TENSORRT_HOME \
to the root of the TensorRT installation."
  exit 1
fi

if [ -f "$SIRIUS_TENSORRT_HOME/bin/trtexec" ]; then
  echo "TensorRT executable found at: $SIRIUS_TENSORRT_HOME/bin/trtexec"
else
  echo "TensorRT executable not found at: $SIRIUS_TENSORRT_HOME/bin/trtexec"
  exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# echo "Script dir: $SCRIPT_DIR"

models=(
  "densenet161"
  "distilbert_base"
  "distilgpt2"
  "efficientnet_v2_s"
  "efficientvit_b2"
  "resnet152"
)

# TMP_LOG=/tmp/sirius-triton-trt-model-log

for model in "${models[@]}"; do
  cmd="python $SCRIPT_DIR/$model.py"
  echo "Preparing model: $model ($cmd)"

  cmd="$cmd"
  eval $cmd

  if [ $? -ne 0 ]; then
    echo "Failed to compile model: $model"
    exit 1
  fi
done

echo "All models compiled."