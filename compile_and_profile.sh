#!/bin/bash

# Compile step
cd hopper && python setup.py install > compile.logs && python demangle_logs.py && cd ..

python test_lite_attention.py

# Profile step
ncu -o bf16_fp8_int8_FA3_LA_profile%i --kernel-name device_kernel --launch-skip 4 --set full python bf16_fp8_int8_FA3_LA_profile.py
