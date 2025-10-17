import torch, argparse, gc, os
from diffusers.utils import export_to_video
from .modify_model.modify_wan import WanTransformer3DModel, set_pv_tune_wan
from .modify_model.modify_wanpipeline import WanPipeline
from .utils import _parse_float_list

os.environ["TOKENIZERS_PARALLELISM"]="false"
prompt = "A vibrant scene of a snowy mountain landscape. The sky is filled with a multitude of colorful hot air balloons, each floating at different heights, creating a dynamic and lively atmosphere. The balloons are scattered across the sky, some closer to the viewer, others further away, adding depth to the scene.  Below, the mountainous terrain is blanketed in a thick layer of snow, with a few patches of bare earth visible here and there. The snow-covered mountains provide a stark contrast to the colorful balloons, enhancing the visual appeal of the scene.  In the foreground, a few cars can be seen driving along a winding road that cuts through the mountains. The cars are small compared to the vastness of the landscape, emphasizing the grandeur of the surroundings.  The overall style of the video is a mix of adventure and tranquility, with the hot air balloons adding a touch of whimsy to the otherwise serene mountain landscape. The video is likely shot during the day, as the lighting is bright and even, casting soft shadows on the snow-covered mountains."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

def parse_args():
    parser = argparse.ArgumentParser(description="Wan Evaluation")
    parser.add_argument(
        "--model-size",
        choices=["1_3b", "14b"],
        default="1_3b",
        help="Wan2.1-T2V model size",
    )
    parser.add_argument("--pv-threshold", type=float, default=20.0)
    parser.add_argument(
        "--mode",
        choices=["tune", "load", "fixed"],
        default="fixed",
        help="Run mode: 'tune' to tune per-timestep pv, 'load' to use tuned JSON, 'fixed' to use --pv-threshold",
    )
    parser.add_argument("--pv-l1-list", type=str, default=None,
                        help="Per-timestep L1 targets for tuning. "
                             "Comma-separated or JSON list, e.g. '0.02,0.03' or '[0.02,0.03]'.")
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    pv_l1 = _parse_float_list(args.pv_l1_list) if args.mode == "tune" else None
    os.environ["model_size"] = args.model_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_size == "1_3b": model_id = "/root/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    else: model_id = "Wan2.1-T2V-14B-Diffusers location"
    # model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    transformer = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    set_pv_tune_wan(
        transformer,
        verbose=args.verbose,
        pv_threshold=args.pv_threshold
    )
    pipe = WanPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(device)

    pipe.enable_model_cpu_offload()

    with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=5.0,
            generator=torch.Generator(device=device).manual_seed(42),
            mode=args.mode,
            pv_l1_list=pv_l1
        ).frames[0]
        if args.mode != "tune": export_to_video(output, f"calibrate/videos/wan2_1-{args.model_size}.mp4", fps=15)
        del output
        gc.collect()
        torch.cuda.empty_cache()