# LiteAttention

LiteAttention is a lightweight Flash Attention 3 wrapper with skip list optimization.

## Requirements
- H100 / H800 GPU
- CUDA >= 12.8
- CUDA toolkit
- PyTorch 2.2 and above
- `packaging` Python package (`pip install packaging`)
- `ninja` Python package (`pip install ninja`) *
- Linux

\* Make sure that `ninja` is installed and that it works correctly (e.g. `ninja
--version` then `echo $?` should return exit code 0). If not (sometimes `ninja
--version` then `echo $?` returns a nonzero exit code), uninstall then reinstall
`ninja` (`pip uninstall -y ninja && pip install ninja`). Without `ninja`,
compiling can take a very long time (2h) since it does not use multiple CPU
cores. With `ninja` compiling takes 3-5 minutes on a 64-core machine using CUDA toolkit.

## Installation

Clone this repo and build from source

```sh
cd hopper
python setup.py install
```

If your machine has less than 96GB of RAM and lots of CPU cores, `ninja` might
run too many parallel compilation jobs that could exhaust the amount of RAM. To
limit the number of parallel compilation jobs, you can set the environment
variable `MAX_JOBS`:
```sh
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

## How to use LiteAttention

```python
from lite_attention import LiteAttention, 


# In your model, set the attention class to be LiteAttention with an optional threshold
# NOTE: `threshold` should be a negative number where the lower it is the less skipping there is.
# NOTE: If a positive number is given, it will be converted to negative (for example 10 ==> -10)
self.attn = LiteAttention(threshold=-6.0)
.
.
.
hidden_states_a_raw = self.attn(query, key, value, scale)

# If you don't know the threshold at the point of initialization, you can set it later via the set_threshold function
self.attn = LiteAttention()
.
.
.
self.attn.set_threshold(threshold=calculated_threshold)

# Additionally, we provide the capability to reset the skip state if needed 
self.attn.reset_skip_state()

# or to toggle the skipping optimization; turning it off falls back to regular FA3
self.attn.enable_skip_optimization(enable=False)
```

## License

LiteAttention is build on top of FA3 which has a BSD 3-Clause license. As such the original code maintains that license and any new code for LiteAttention is distributed under an MIT license.

See [LICENSE-BSD](LICENSE-BSD) and [LICENSE-MIT](LICENSE-MIT) for further details.
