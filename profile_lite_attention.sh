# ncu -set full -f lite_attention%i.ncu-rep ./test_lite_attention.py

ncu \
  --set full \
  -o lite_attention%i \
  --kernel-name device_kernel \
  python test_lite_attention.py