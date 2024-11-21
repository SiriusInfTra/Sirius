SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# echo "Script dir: $SCRIPT_DIR"

tune_models=(
  "distilbert-base"
  "distilgpt2"
  "efficientnet_v2_s"
  "efficientvit_b2"
)

models=(
  "densenet161"
  "resnet152"
)

TMP_RECORD_PREFIX=/tmp/colsys-tvm-tune-record

for model in "${tune_models[@]}"; do
  cmd="python $SCRIPT_DIR/$model.py --tune --record $TMP_RECORD_PREFIX-$model"
  echo "Tuning model: $model ($cmd)"
  eval $cmd
  if [ $? -ne 0 ]; then
    echo "Failed to tune model: $model"
    exit 1
  fi

  echo "Preapre model: $model ($cmd)"
  cmd="python $SCRIPT_DIR/$model.py --record $TMP_RECORD_PREFIX-$model"
  eval $cmd
  if [ $? -ne 0 ]; then
    echo "Failed to compile model: $model"
    exit 1
  fi
done

for model in "${models[@]}"; do
  cmd="python $SCRIPT_DIR/$model.py"
  echo "Preparing model: $model ($cmd)"
  eval $cmd
  if [ $? -ne 0 ]; then
    echo "Failed to compile model: $model"
    exit 1
  fi
done