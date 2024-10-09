#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "\$0")")
mkdir -p "$SCRIPT_DIR/build"
cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" -DCMAKE_BUILD_TYPE=Release

make -C "$SCRIPT_DIR/build" -j32
