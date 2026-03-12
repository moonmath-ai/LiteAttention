#!/bin/bash

# Compile step
python setup.py install > compile.logs && python hopper/demangle_logs.py

python hopper/tests/test_lite_attention.py

# Profile step
ncu -o bf16_fp8_int8_FA3_LA_profile%i --kernel-name device_kernel --launch-skip 4 --set full python bf16_fp8_int8_FA3_LA_profile.py
