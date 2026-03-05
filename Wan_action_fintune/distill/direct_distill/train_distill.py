#!/usr/bin/env python3
"""
Direct distill training for Wan video model.
Uses CSV + video dataset (same input format as inference: prompt, input_image, action_seq)
plus columns: video (teacher output), seed, rand_device, num_inference_steps, cfg_scale.
"""
import os
import sys
import argparse

# Add Wan_action_fintune to path so we can import train.train
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_wan_fintune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _wan_fintune_root not in sys.path:
    sys.path.insert(0, _wan_fintune_root)

import torch
import accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import (
    ToAbsolutePath,
    LoadImage,
    LoadVideo,
    LoadActionSequence,
    ImageCropAndResize,
    RouteByExtensionName,
    DataProcessingOperator,
)
from diffsynth.diffusion import (
    add_general_config,
    add_video_size_config,
    launch_training_task,
    ModelLogger,
)
from diffsynth.diffusion.parsers import add_gradient_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FirstElement(DataProcessingOperator):
    """Take first element if data is list/tuple (e.g. LoadVideo returns list of 1 frame)."""

    def __call__(self, data):
        if isinstance(data, (list, tuple)) and len(data) > 0:
            return data[0]
        return data


def _input_image_operator(base_path, height, width, num_frames, max_pixels=1920 * 1080):
    """Load input_image: single image file -> PIL; video file -> first frame (aligned with inference)."""
    crop_resize = ImageCropAndResize(
        height=height,
        width=width,
        max_pixels=max_pixels,
        height_division_factor=16,
        width_division_factor=16,
    )
    load_video_first_frame = LoadVideo(
        num_frames=1,
        time_division_factor=1,
        time_division_remainder=0,
        frame_processor=crop_resize,
    ) >> FirstElement()
    return (
        ToAbsolutePath(base_path)
        >> RouteByExtensionName(
            [
                (("jpg", "jpeg", "png", "webp", "bmp"), LoadImage()),
                (
                    ("mp4", "avi", "mov", "mkv", "webm", "wmv", "flv"),
                    load_video_first_frame,
                ),
            ]
        )
    )


def create_dataset(args):
    """Build UnifiedDataset for direct_distill: video (teacher) + input_image + action_seq (same as inference)."""
    main_operator = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=getattr(args, "max_pixels", 1920 * 1080),
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=args.num_frames,
        time_division_factor=4,
        time_division_remainder=1,
    )
    special_operator_map = {
        "input_image": _input_image_operator(
            args.dataset_base_path,
            args.height,
            args.width,
            args.num_frames,
        ),
        "action_seq": ToAbsolutePath(args.dataset_base_path)
        >> LoadActionSequence(joint_dim=args.action_joint_dim),
    }
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=["video", "input_image", "action_seq"],
        main_data_operator=main_operator,
        special_operator_map=special_operator_map,
    )


def parser():
    p = argparse.ArgumentParser(description="Wan video direct distill (CSV + video, aligned with inference).")
    p.add_argument("--dataset_base_path", type=str, required=True, help="Root dir for paths in CSV (or absolute paths).")
    p.add_argument("--dataset_metadata_path", type=str, required=True, help="CSV: prompt,input_image,action_seq,video,seed,rand_device,num_inference_steps,cfg_scale.")
    p.add_argument("--dataset_repeat", type=int, default=1)
    p.add_argument("--dataset_num_workers", type=int, default=0)
    p.add_argument("--height", type=int, default=240, help="Same as inference default.")
    p.add_argument("--width", type=int, default=320, help="Same as inference default.")
    p.add_argument("--num_frames", type=int, default=17, help="Same as inference default.")
    p.add_argument("--max_pixels", type=int, default=1920 * 1080)
    p.add_argument("--model_paths", type=str, default=None, help="JSON list of local ckpt paths (same layout as inference model_dir).")
    p.add_argument("--model_id_with_origin_paths", type=str, default=None, help="ModelScope model_id:pattern, comma-separated.")
    p.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer dir (e.g. {model_dir}/google/umt5-xxl).")
    p.add_argument("--extra_inputs", type=str, default="input_image,action_seq,seed,rand_device,num_inference_steps,cfg_scale")
    p.add_argument("--trainable_models", type=str, default="dit")
    p.add_argument("--lora_base_model", type=str, default=None)
    p.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_checkpoint", type=str, default=None)
    p.add_argument("--preset_lora_path", type=str, default=None)
    p.add_argument("--preset_lora_model", type=str, default=None)
    p.add_argument("--task", type=str, default="direct_distill")
    p.add_argument("--output_path", type=str, default="./models/train/wan_direct_distill")
    p.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.")
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--action_joint_dim", type=int, default=14, help="Same as inference joint_dim.")
    p.add_argument("--max_timestep_boundary", type=float, default=1.0)
    p.add_argument("--min_timestep_boundary", type=float, default=0.0)
    p.add_argument("--initialize_model_on_cpu", action="store_true")
    p.add_argument("--find_unused_parameters", action="store_true")
    add_gradient_config(p)
    return p


def main():
    args = parser().parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )

    dataset = create_dataset(args)

    from train.train import WanTrainingModule

    # Use data["input_image"] when present (from CSV column), else data["video"][0]
    class WanDistillTrainingModule(WanTrainingModule):
        def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
            for extra_input in extra_inputs:
                if extra_input == "input_image":
                    img = data.get("input_image")
                    if img is not None:
                        inputs_shared["input_image"] = img[0] if isinstance(img, (list, tuple)) else img
                    else:
                        inputs_shared["input_image"] = data["video"][0]
                elif extra_input == "end_image":
                    inputs_shared["end_image"] = data["video"][-1]
                elif extra_input in ("reference_image", "vace_reference_image"):
                    inputs_shared[extra_input] = data[extra_input][0]
                elif extra_input == "action_seq":
                    inputs_shared["action_seq"] = data["action_seq"]
                else:
                    inputs_shared[extra_input] = data[extra_input]
            return inputs_shared

    model = WanDistillTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=None,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=getattr(args, "use_gradient_checkpointing_offload", False),
        extra_inputs=args.extra_inputs,
        fp8_models=None,
        offload_models=None,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        action_joint_dim=args.action_joint_dim,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launch_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.dataset_num_workers,
        save_steps=args.save_steps,
        num_epochs=args.num_epochs,
        args=args,
    )


if __name__ == "__main__":
    main()
