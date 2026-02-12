import torch, os, argparse, accelerate, warnings, glob
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.core.data.parquet_streaming_dataset import (
    ParquetStreamingDataset,
    collate_robot_batch,
)
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(tokenizer_path) if tokenizer_path is not None else None
        audio_processor_config = ModelConfig(audio_processor_path) if audio_processor_path is not None else None
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "action_seq":  
                inputs_shared["action_seq"] = data["action_seq"]  
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--action_joint_dim", type=int, default=None, help="Dimension of the action sequence vectors (D). If not provided, it will be inferred.")
    # Parquet streaming dataloader parameters
    parser.add_argument("--parquet_dir", type=str, required=True, help="Directory containing Parquet shard files.")
    parser.add_argument("--window_stride", type=int, default=1, help="Stride for sliding window sampling (default: 1, maximum overlap).")
    parser.add_argument("--shuffle_buffer_size", type=int, default=1000, help="Size of shuffle buffer for streaming randomization (default: 1000).")
    parser.add_argument("--dataloader_seed", type=int, default=None, help="Random seed for dataloader shuffling.")
    # TensorBoard parameters
    parser.add_argument("--enable_tensorboard", default=True, action="store_true", help="Enable TensorBoard logging.")
    parser.add_argument("--disable_tensorboard", dest="enable_tensorboard", action="store_false", help="Disable TensorBoard logging.")
    parser.add_argument("--tensorboard_log_interval", type=int, default=10, help="Log metrics to TensorBoard every N steps (default: 10).")
    return parser


def create_dataset(args):
    """
    Create Parquet streaming dataset for action-conditioned Wan model training.
    
    Args:
        args: Command-line arguments
    
    Returns:
        ParquetStreamingDataset instance
    """
    # Find Parquet shard files
    parquet_paths = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    
    if not parquet_paths:
        raise ValueError(f"No Parquet files found in {args.parquet_dir}")
    
    print(f"Using Parquet streaming dataloader:")
    print(f"  Shards: {len(parquet_paths)}")
    print(f"  Window size: {args.num_frames} frames")
    print(f"  Window stride: {args.window_stride}")
    print(f"  Shuffle buffer: {args.shuffle_buffer_size}")
    
    # Create frame transform
    frame_transform = ImageCropAndResize(
        height=args.height,
        width=args.width,
        max_pixels=args.max_pixels,
        height_division_factor=16,
        width_division_factor=16,
    )
    
    # Create streaming dataset
    dataset = ParquetStreamingDataset(
        parquet_paths=parquet_paths,
        window_size=args.num_frames,
        window_stride=args.window_stride,
        shuffle_buffer_size=args.shuffle_buffer_size,
        frame_transform=frame_transform,
        action_dim=args.action_joint_dim,
        seed=args.dataloader_seed,
    )
    
    return dataset


def launch_streaming_training_task(
    accelerator: accelerate.Accelerator,
    dataset: torch.utils.data.IterableDataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    """
    Training launcher optimized for streaming (IterableDataset) datasets.
    
    Key differences from standard launcher:
    - No shuffle (streaming handles its own randomization)
    - Uses collate_robot_batch for proper batching
    - Supports prefetching and persistent workers
    """
    from tqdm import tqdm
    
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    # Create dataloader for streaming dataset
    # Note: shuffle=False for IterableDataset (it handles its own shuffling)
    dataloader_kwargs = {
        "batch_size": 1,  # Each sample is already a window
        "num_workers": num_workers,
        "collate_fn": lambda batch: batch[0],  # Unwrap single item
        "pin_memory": True,
    }
    
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
        dataloader_kwargs["persistent_workers"] = True
    
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    
    # Note: Do NOT prepare the dataloader with accelerator.
    # The IterableDataset handles its own shard partitioning, and accelerate's
    # prepare() would try to concatenate PIL Images in the batch, causing:
    #   TypeError: Can only concatenate tensors but got <class 'PIL.Image.Image'>
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader, desc=f"Epoch {epoch_id + 1}/{num_epochs}", total=len(dataset)):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, optimizer=optimizer, batch_size=1)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
    model_logger.close()


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    
    # Create Parquet streaming dataset
    dataset = create_dataset(args)
    
    # Create model
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    
    # Create model logger
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard=args.enable_tensorboard,
        log_interval=args.tensorboard_log_interval,
    )
    
    # Launch training with streaming dataloader
    supported_tasks = ["sft", "sft:train", "direct_distill", "direct_distill:train"]
    if args.task not in supported_tasks:
        raise ValueError(f"Task '{args.task}' is not supported. Supported tasks: {supported_tasks}")
    
    launch_streaming_training_task(accelerator, dataset, model, model_logger, args=args)
