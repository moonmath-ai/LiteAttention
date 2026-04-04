#!/bin/bash

# Compile and run with ReInt8 = true
echo "Compiling and running with ReInt8 = true..."
nvcc -std=c++20 --use_fast_math -I./csrc/cutlass/include -arch=sm_90a -DREINT8_ENABLED -o reint8 reint8.cu && \
./reint8 > shapes_and_such.txt 2>&1

if [ $? -eq 0 ]; then
    echo "✓ ReInt8 = true completed successfully"
else
    echo "✗ ReInt8 = true failed"
    exit 1
fi

# Compile and run with ReInt8 = false
echo "Compiling and running with ReInt8 = false..."
nvcc -std=c++20 --use_fast_math -I./csrc/cutlass/include -arch=sm_90a -o reint8 reint8.cu && \
./reint8 > shapes_and_such_no_reint8.txt 2>&1

if [ $? -eq 0 ]; then
    echo "✓ ReInt8 = false completed successfully"
else
    echo "✗ ReInt8 = false failed"
    exit 1
fi

echo "All runs completed successfully!"

