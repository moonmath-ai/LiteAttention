import torch, json
import argparse
from .modify_model.modify_ltxpipeline import LTXConditionPipeline
from diffusers import LTXLatentUpsamplePipeline
from .modify_model.modify_ltx import LTXVideoTransformer3DModel, set_pv_tune_ltx, reset_pv_tune_ltx
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video
from utils import _parse_float_list

def round_to_nearest_resolution_acceptable_by_vae(height, width, pipe):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pv-threshold-1", type=float, default=20.0)
    parser.add_argument("--pv-threshold-2", type=float, default=20.0)
    parser.add_argument(
        "--gen-mode",
        choices=["tune", "load", "fixed"],
        default="fixed",
        help="Generate stage: 'tune' to tune per-timestep pv, 'load' to use tuned JSON, 'fixed' to use --pv-threshold-1",
    )
    parser.add_argument(
        "--den-mode",
        choices=["tune", "load", "fixed"],
        default="fixed",
        help="Denoise stage: 'tune' to tune per-timestep pv, 'load' to use tuned JSON, 'fixed' to use --pv-threshold-2",
    )
    # parser.add_argument("--pv-l1-gen", type=float, default=0.05,
    #                 help="Base L1 target for generate stage (middle bucket).")
    # parser.add_argument("--pv-l1-den", type=float, default=0.05,
    #                     help="Base L1 target for denoise stage (middle bucket).")
    parser.add_argument("--pv-l1-gen-list", type=str, default=None,
                        help="Per-timestep L1 targets for tuning the generate stage. "
                             "Comma-separated or JSON list, e.g. '0.02,0.03' or '[0.02,0.03]'.")
    parser.add_argument("--pv-l1-den-list", type=str, default=None,
                        help="Per-timestep L1 targets for tuning the denoise stage. "
                             "Comma-separated or JSON list, e.g. '0.03,0.035' or '[0.03,0.035]'.")
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    args = parser.parse_args()

    # Fixed PV values (used when mode == 'fixed') and Mode
    pv1 = args.pv_threshold_1
    pv2 = args.pv_threshold_2
    gen_mode = args.gen_mode
    den_mode = args.den_mode
    pv_l1_gen_list = _parse_float_list(args.pv_l1_gen_list)
    pv_l1_den_list = _parse_float_list(args.pv_l1_den_list)

    # Load models
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    # default pv for processors (used only when not loading per-step from JSON)
    set_pv_tune_ltx(transformer, verbose=args.verbose, pv_threshold=pv1, part="generate")
    pipe = LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        "Lightricks/ltxv-spatial-upscaler-0.9.7",
        vae=pipe.vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    pipe_upsample.enable_model_cpu_offload()

    # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png")
    image = load_image("calibrate/videos/penguin.png")
    video = load_video(export_to_video([image]))  # compress the image using video compression as the model was trained on videos
    condition1 = LTXVideoCondition(video=video, frame_index=0)

    prompt = "A cute little penguin takes out a book and starts reading it"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    expected_height, expected_width = 480, 832
    downscale_factor = 2 / 3
    num_frames = 96

    # Part 1. Generate video at smaller resolution
    downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width, pipe)
    latents = pipe(
        conditions=[condition1],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=num_frames,
        num_inference_steps=30,
        generator=torch.Generator().manual_seed(0),
        output_type="latent",  # Crucial: ask the pipeline to return latents, not decoded frames
        mode=gen_mode,
        part="generate",
        pv_l1_list=pv_l1_gen_list
    ).frames

    # Part 2. Upscale generated video using latent upsampler with fewer inference steps
    # The available latent upsampler upscales the height/width by 2x
    upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
    upscaled_latents = pipe_upsample(
        latents=latents,
        output_type="latent"
    ).frames

    # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
    reset_pv_tune_ltx(transformer, pv_threshold=pv2, part="denoise")
    video = pipe(
        conditions=[condition1],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=upscaled_width,
        height=upscaled_height,
        num_frames=num_frames,
        denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
        num_inference_steps=10,
        latents=upscaled_latents,
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
        generator=torch.Generator().manual_seed(0),
        output_type="pil",
        mode=den_mode,
        part="denoise",
        pv_l1_list=pv_l1_den_list
    ).frames[0]

    if gen_mode != "tune" and den_mode != "tune": # inference mode
        # Part 4. Downscale the video to the expected resolution
        video = [frame.resize((expected_width, expected_height)) for frame in video]
        export_to_video(video, f"./calibrate/videos/ltx.mp4", fps=24)
    else:
        del video

if __name__ == "__main__":
    main()
