#!/bin/bash

# Compile step
cd hopper && python setup.py install > ../reint8/compile.logs && python ../reint8/demangle_logs.py && cd ..

# python test_lite_attention.py
# python ./reint8/events_bf16_fp8_int8_FA3_LA_profile.py

# Profile step
ncu -o profile/bf16_fp8_int8_FA3_LA_profile%i --kernel-name device_kernel --launch-skip 4 --set full --section-folder-recursive ../Sections python reint8/bf16_fp8_int8_FA3_LA_profile.py
# ncu -o profile/bf16_fp8_int8_FA3_LA_profile%i --kernel-name device_kernel --launch-skip 4 --set full python reint8/bf16_fp8_int8_FA3_LA_profile.py
