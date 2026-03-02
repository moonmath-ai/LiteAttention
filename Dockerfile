FROM timuringo/flash_attn3:latest

WORKDIR /lite-attention
COPY . .
RUN LITE_ATTENTION_FORCE_CXX11_ABI=FALSE pip install .[dev] --no-build-isolation