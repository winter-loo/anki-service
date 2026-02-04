#!/bin/bash

set -e

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $ROOT_DIR/..

if [ "$BUILD_ROOT" == "" ]; then
    out=$(pwd)/out
else
    out="$BUILD_ROOT"
fi
export CARGO_TARGET_DIR=$out/rust
export RECONFIGURE_KEY="${MAC_X86};${SOURCEMAP}"

# separate build+run steps so build env doesn't leak into subprocesses
cargo build -p runner
exec $out/rust/debug/runner build $*
