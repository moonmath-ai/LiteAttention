# PV Calibration
Per-timestep **pv_threshold** calibration for multimodal diffusion models via skip-PV attention.

Models supported:
- [LTX-Video](https://huggingface.co/Lightricks/LTX-Video-0.9.7-dev)
- [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers), [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)
- Wan2.2 (TO DO)

## Installation
```bash
pip install -r requirements.txt
```

## Usage Examples
### LTX
💡 Model path: In [`ltx_example.py`](./calibrate/ltx_example.py), modify `from_pretrained("Lightricks/LTX-Video-0.9.7-dev", ...)` to your local weights directory.

**Tuning:**
```bash
# Tune generate (denoise fixed at pv=2)
python -m calibrate.ltx_example \
  --gen-mode tune \
  --pv-l1-gen-list '[0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075]' \
  --den-mode fixed \
  --pv-threshold-2 2

# Tune denoise (generate fixed at pv=3.5)
python -m calibrate.ltx_example \
  --gen-mode fixed \
  --pv-threshold-1 3.5 \
  --den-mode tune \
  --pv-l1-den-list '[0.35,0.36,0.37,0.37]'
```
`--pv-l1-*-list` accepts a JSON list, allowing flexible setting of pv_l1 bounds for each timestep during tuning. Its length must match the total timesteps, which in our example are:
- generate: `num_inference_steps=30`
- denoise: `denoise_strength=0.4` * `num_inference_steps=10` = 4

Tuned thresholds are stored in `./calibrate/models_dict/model_name/part_name.json`.

**Inference:**
```bash
# Load tuned thresholds for both parts
python -m calibrate.ltx_example --gen-mode load --den-mode load

# Fixed thresholds
python -m calibrate.ltx_example --gen-mode fixed --den-mode fixed --pv-threshold-1 3.5 --pv-threshold-2 2
```
Final generated video: `calibrate/videos/model_name.mp4`.

### Wan2.1
💡 Model path: In [`wan_example.py`](./calibrate/wan_example.py), modify `model_id` to your local path. This code supports both `Wan2.1-T2V-1.3B-Diffusers` and `Wan2.1-T2V-14B-Diffusers`, set via the command line `--model-size` parameter.

**Tuning:**
```bash
python -m calibrate.wan_example \
  --model-size 1_3b \
  --mode tune \
  --pv-l1-list '[0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075]'
```

**Inference:**
```bash
# Load tuned thresholds
python -m calibrate.wan_example --model-size 1_3b --mode load

# Fixed thresholds
python -m calibrate.wan_example --model-size 1_3b --mode fixed --pv-threshold 3
```