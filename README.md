# Custormized Flash Attention for ROCM
## Requirements
- GPU: AMD MI250 arch:gfx90a  
- OS: Ubuntu 20.04  
- ROCM: 5.4  
- Software: pytorch-rocm 2.0 python 3.8

## Step
- bash 0-rebuild-fmha_api.sh # No need for this if you have python 3.8
- pip install .

## How to use or test
Refer: flash_attn_rocm/test_rocm.py
