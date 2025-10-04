#!/bin/bash

# Flash Attention Compile Script
# This script sets environment variables to disable unused features and compiles flash_attn_3

echo "Setting Flash Attention environment variables..."

# In order as they appear in setup.py (lines 47-68)
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE      # Line 47 - No training

export FLASH_ATTENTION_DISABLE_SPLIT=TRUE       # Line 48 - If sequences fit in memory  
# export FLASH_ATTENTION_DISABLE_SPLIT=FALSE        # Line 48 - If sequences fit in memory  

export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE       # Line 49 - Not needed without autoregressive
# export FLASH_ATTENTION_DISABLE_PAGEDKV=FALSE    # Line 49 - Not needed without autoregressive

export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE      # Line 50 - Not needed without autoregressive
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE         # Line 51 - If no sliding window attention
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE       # Line 52 - If no attention softcapping

export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE       # Line 53 - If not using Grouped Query Attention
# export FLASH_ATTENTION_DISABLE_PACKGQA=FALSE    # Line 53 - If not using Grouped Query Attention

export FLASH_ATTENTION_DISABLE_FP16=TRUE          # Line 54 - Only using bf16
export FLASH_ATTENTION_DISABLE_FP8=TRUE           # Line 55 - Only using bf16

export FLASH_ATTENTION_DISABLE_VARLEN=TRUE        # Line 56 - Be careful, might need this for batching
# export FLASH_ATTENTION_DISABLE_VARLEN=FALSE     # Line 56 - Be careful, might need this for batching

export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE       # Line 57 - Usually not needed
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE        # Line 58 - Don't need 64 (you use 128)
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE        # Line 59 - Don't need 96 (you use 128)
# export FLASH_ATTENTION_DISABLE_HDIM128=TRUE     # Line 60 - DON'T disable this (you need 128)
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE       # Line 61 - Don't need 192 (you use 128)

export FLASH_ATTENTION_DISABLE_HDIM256=TRUE       # Line 62 - Don't need 256 (you use 128)
# export FLASH_ATTENTION_DISABLE_HDIM256=FALSE    # Line 62 - Don't need 256 (you use 128)

export FLASH_ATTENTION_DISABLE_SM80=TRUE          # Line 63 - H100 only (SM90)
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE    # Line 67 - Disable different V head dims
export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE   # Line 68 - Disable different V head dims

echo "Environment variables set successfully!"

# Activate conda environment
# echo "Activating conda environment ltxfa3_comp..."
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ltxfa3_comp

# Uninstall existing flash_attn_3 installation
echo "Uninstalling existing flash_attn_3..."
pip uninstall -y flash_attn_3

# Change to hopper directory
echo "Changing to hopper directory..."
cd hopper

# Install flash_attn_3
echo "Starting flash_attn_3 installation..."
python setup.py install

echo "Compilation complete!"
