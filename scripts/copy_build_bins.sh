copy_bin_from_to() {
  local from=$1
  local to=$2

  for file in $(ls $from); do

  if [[ -d "$from/$file" ]]; then
    copy_bin_from_to "$from/$file" "$to/$file"
  elif [[ -f "$from/$file" && -x "$from/$file" ]]; then
    mkdir -p "$to"
    cp "$from/$file" "$to/$file"
  fi
  done
}

rm -rf ./bin
copy_bin_from_to ./build ./bin

mkdir -p ./bin/xsched
cp -r ./build/xsched/lib ./bin/xsched/lib