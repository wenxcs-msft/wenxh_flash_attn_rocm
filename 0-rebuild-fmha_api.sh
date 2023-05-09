#!/bin/bash
declare CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
rm $CWD/flash_attn_rocm/libfmha_api.so
rm -rf $CWD/build
cmake -B $CWD/build -S $CWD/csrc
cmake --build $CWD/build
cp $CWD/build/libfmha_api.so $CWD/flash_attn_rocm/libfmha_api.so