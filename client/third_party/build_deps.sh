#!/bin/bash
# 获取脚本的完整路径
SCRIPT_PATH=$(readlink -f "$0")

# 获取脚本所在的目录
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

mkdir -p "$SCRIPT_DIR/build"
cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" -DCMAKE_BUILD_TYPE=Release

make -C "$SCRIPT_DIR/build" -j32

