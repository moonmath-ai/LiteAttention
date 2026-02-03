#!/bin/bash

# Compile step
cd hopper && python setup.py install > ../reint8/compile.logs && cd ../reint8 && python demangle_logs.py && cd ..

python test_lite_attention.py
# python reint8/events_bf16_fp8_int8_FA3_LA_profile.py

# sudo nvidia-modprobe -m -u

# # Ensure profile directory exists
# mkdir -p reint8/profile

# # Get absolute path to profile directory
# PROFILE_DIR="$(cd "$(dirname "$0")" && pwd)/profile"

# # Profile step
# # ncu -o profile/bf16_fp8_int8_FA3_LA_profile%i --kernel-name device_kernel --launch-skip 4 --set full --section-folder-recursive ../Sections python reint8/bf16_fp8_int8_FA3_LA_profile.py
# # Run ncu with sudo to access GPU performance counters, preserving environment
# # Use absolute path for output to avoid permission issues
# # Set HOME and USER to current user to avoid issues with sudo environment
# sudo -E env "PATH=$PATH" "HOME=$HOME" "USER=$USER" ncu -o "${PROFILE_DIR}/bf16_fp8_int8_FA3_LA_profile%i" --kernel-name device_kernel --launch-skip 4 --set full python reint8/bf16_fp8_int8_FA3_LA_profile.py
