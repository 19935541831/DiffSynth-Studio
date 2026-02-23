"""
Parquet Streaming Dataset for Robot Trajectory Data

This module provides a high-efficiency streaming dataloader for robot training data
stored in Parquet format. Key features:

1. True streaming with IterableDataset - no need to load all indices
2. Sliding window on-the-fly - generates windows from streaming frames
3. Shuffle buffer - randomizes data in streaming mode
4. Multi-worker support - automatic shard partitioning
5. GPU-accelerated frame decoding (optional)

Usage:
    dataset = ParquetStreamingDataset(
        parquet_paths=glob.glob("data/*.parquet"),
        window_size=17,
        window_stride=1,
        shuffle_buffer_size=10000,
        frame_transform=ImageCropAndResize(height=240, width=320),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        collate_fn=collate_robot_batch,
    )
"""

import io
import os
import random
import time
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple
import pyarrow.parquet as pq
import pyarrow.compute as pc

class SlidingWindowIterator:
    """
    Generate sliding windows from streaming trajectory data.
    
    Handles episode boundaries by clearing the buffer when a new episode starts.
    Emits windows according to the specified stride.
    
    Example:
        iterator = SlidingWindowIterator(window_size=17, stride=1)
        for frame_data, action, episode_id in stream:
            window = iterator.add_frame(frame_data, action, frame_idx, episode_id, instruction, task_name)
            if window is not None:
                yield window
    """
    
    def __init__(self, window_size: int, stride: int = 1):
        """
        Initialize the sliding window iterator.
        
        Args:
            window_size: Number of frames per window
            stride: Number of frames to advance between windows
        """
        self.window_size = window_size
        self.stride = stride
        
        # Frame buffer: stores (frame_data, action) tuples
        self.buffer: deque = deque(maxlen=window_size)
        
        # Metadata for current trajectory
        self.current_episode_id: Optional[str] = None
        self.current_instruction: Optional[str] = None
        self.current_task_name: Optional[str] = None
        
        # Counter for stride-based emission
        self.frames_since_emit: int = 0
    
    def add_frame(
        self,
        frame_data: bytes,
        action: List[float],
        frame_idx: int,
        episode_id: str,
        instruction: str,
        task_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Add a frame to the buffer and potentially emit a window.
        
        Args:
            frame_data: JPEG-compressed frame bytes
            action: Action vector as list of floats
            frame_idx: Frame index within the episode
            episode_id: Current episode identifier
            instruction: Instruction text for this episode
            task_name: Task name for this episode
        
        Returns:
            Window dict if ready to emit, None otherwise
        """
        # Handle episode boundary
        if episode_id != self.current_episode_id:
            self._reset_for_new_episode(episode_id, instruction, task_name)
        
        # Add frame to buffer
        self.buffer.append((frame_data, action, frame_idx))
        self.frames_since_emit += 1
        
        # Check if we can emit a window
        if len(self.buffer) == self.window_size:
            if self.frames_since_emit >= self.stride:
                self.frames_since_emit = 0
                return self._emit_window()
        
        return None
    
    def _reset_for_new_episode(
        self,
        episode_id: str,
        instruction: str,
        task_name: str,
    ):
        """Reset state for a new episode."""
        self.buffer.clear()
        self.frames_since_emit = 0
        self.current_episode_id = episode_id
        self.current_instruction = instruction
        self.current_task_name = task_name
    
    def _emit_window(self) -> Dict[str, Any]:
        """Create and return a window from the current buffer."""
        frames_data = [item[0] for item in self.buffer]
        actions = [item[1] for item in self.buffer]
        frame_indices = [item[2] for item in self.buffer]
        
        return {
            "frames_data": frames_data,  # List of JPEG bytes
            "actions": actions,           # List of action vectors
            "frame_indices": frame_indices,  # List of frame indices
            "instruction": self.current_instruction,
            "task_name": self.current_task_name,
            "episode_id": self.current_episode_id,
        }
    
    def flush(self) -> Optional[Dict[str, Any]]:
        """
        Flush any remaining complete window at end of stream.
        
        Returns:
            Final window if buffer is full, None otherwise
        """
        if len(self.buffer) == self.window_size:
            return self._emit_window()
        return None
    
    def reset(self):
        """Reset the iterator state completely."""
        self.buffer.clear()
        self.current_episode_id = None
        self.current_instruction = None
        self.current_task_name = None
        self.frames_since_emit = 0


class ShuffleBuffer:
    """
    Reservoir-based shuffle buffer for streaming data randomization.
    
    When the buffer is full, new items randomly replace existing items,
    and the displaced item is returned. This provides approximate shuffling
    within a window of buffer_size items.
    
    Example:
        buffer = ShuffleBuffer(buffer_size=10000)
        for item in stream:
            output = buffer.add_and_sample(item)
            if output is not None:
                yield output
        # At end of stream, flush remaining items
        for item in buffer.flush():
            yield item
    """
    
    def __init__(self, buffer_size: int, seed: Optional[int] = None):
        """
        Initialize the shuffle buffer.
        
        Args:
            buffer_size: Maximum number of items to buffer
            seed: Optional random seed for reproducibility
        """
        self.buffer_size = buffer_size
        self.buffer: List[Any] = []
        
        if seed is not None:
            random.seed(seed)
    
    def add_and_sample(self, item: Any) -> Optional[Any]:
        """
        Add an item and potentially return a randomly displaced item.
        
        Args:
            item: Item to add to the buffer
        
        Returns:
            Displaced item if buffer was full, None otherwise
        """
        if len(self.buffer) < self.buffer_size:
            # Buffer not full yet, just append
            self.buffer.append(item)
            return None
        else:
            # Buffer full - randomly replace and return displaced item
            idx = random.randint(0, self.buffer_size - 1)
            output = self.buffer[idx]
            self.buffer[idx] = item
            return output
    
    def flush(self) -> Iterator[Any]:
        """
        Yield remaining items in shuffled order at end of epoch.
        
        Yields:
            Items from the buffer in random order
        """
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class FrameDecoder:
    """
    Decode JPEG bytes to PIL Images or tensors.
    
    Supports:
        - TurboJPEG decoding (2-6x faster, requires PyTurboJPEG)
        - CPU decoding via PIL (fallback)
        - GPU-accelerated decoding via torchvision (if available)
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize the frame decoder.
        
        Args:
            use_gpu: Whether to attempt GPU decoding
            device: CUDA device for GPU decoding
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = device
        
        # Try to use TurboJPEG for faster CPU decoding (2-6x faster than PIL)
        self._turbo = None
        try:
            from turbojpeg import TurboJPEG
            self._turbo = TurboJPEG()
        except (ImportError, OSError, RuntimeError):
            pass
        
        # Check for GPU decoding support
        self._has_gpu_decode = False
        if self.use_gpu:
            try:
                import torchvision.io
                if hasattr(torchvision.io, 'decode_jpeg'):
                    self._has_gpu_decode = True
            except ImportError:
                pass
    
    def decode(self, jpeg_bytes: bytes) -> Image.Image:
        """
        Decode JPEG bytes to PIL Image.
        
        Args:
            jpeg_bytes: JPEG-compressed bytes
        
        Returns:
            PIL Image in RGB format
        """
        if self._turbo is not None:
            # TurboJPEG decoding (2-6x faster than PIL)
            bgr_array = self._turbo.decode(jpeg_bytes)
            rgb_array = bgr_array[:, :, ::-1].copy()  # BGR -> RGB
            return Image.fromarray(rgb_array)
        else:
            # PIL fallback
            buffer = io.BytesIO(jpeg_bytes)
            image = Image.open(buffer)
            return image.convert("RGB")
    
    def decode_batch(self, jpeg_bytes_list: List[bytes]) -> List[Image.Image]:
        """
        Decode a batch of JPEG bytes to PIL Images.
        
        Uses a thread pool when TurboJPEG is available (it releases the GIL),
        giving ~2-4x speedup for typical window sizes (e.g. 17 frames).
        Falls back to sequential decoding for PIL (GIL-bound).
        
        Args:
            jpeg_bytes_list: List of JPEG-compressed bytes
        
        Returns:
            List of PIL Images in RGB format
        """
        if self._turbo is not None and len(jpeg_bytes_list) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(jpeg_bytes_list))) as pool:
                return list(pool.map(self.decode, jpeg_bytes_list))
        return [self.decode(b) for b in jpeg_bytes_list]
    
    def decode_to_tensor(self, jpeg_bytes: bytes) -> torch.Tensor:
        """
        Decode JPEG bytes to tensor (optionally on GPU).
        
        Args:
            jpeg_bytes: JPEG-compressed bytes
        
        Returns:
            Tensor of shape (C, H, W) in RGB format, values 0-255
        """
        if self._has_gpu_decode:
            import torchvision.io
            byte_tensor = torch.frombuffer(jpeg_bytes, dtype=torch.uint8)
            return torchvision.io.decode_jpeg(byte_tensor, device=self.device)
        else:
            # Fall back to PIL + convert
            image = self.decode(jpeg_bytes)
            return torch.from_numpy(np.array(image)).permute(2, 0, 1)


class ParquetStreamingDataset(torch.utils.data.IterableDataset):
    """
    High-efficiency streaming dataset for robot trajectory data in Parquet format.
    
    Features:
        - True streaming with IterableDataset
        - On-the-fly sliding window generation
        - Shuffle buffer for streaming randomization
        - Automatic multi-worker shard partitioning
        - Full distributed training support (multi-GPU/multi-node)
        - Optional frame transformation
    
    Distributed Training:
        The dataset implements two-level shard partitioning:
        1. First level: Shards are divided among distributed ranks (GPUs/nodes)
        2. Second level: Each rank's shards are further divided among DataLoader workers
        
        Use set_epoch() at the start of each epoch to ensure proper shuffling:
            for epoch in range(num_epochs):
                dataset.set_epoch(epoch)
                for batch in dataloader:
                    ...
    
    Output format per sample:
        {
            "frames": List[PIL.Image],  # Decoded and transformed frames
            "actions": torch.Tensor,     # (window_size, action_dim)
            "instruction": str,          # Task instruction
        }
    """
    
    def __init__(
        self,
        parquet_paths: List[str],
        window_size: int,
        window_stride: int = 1,
        shuffle_buffer_size: int = 1000,
        frame_transform: Optional[Callable] = None,
        action_dim: Optional[int] = None,
        use_gpu_decode: bool = False,
        seed: Optional[int] = None,
        shuffle_shards: bool = True,
        estimate_length: bool = True,
    ):
        """
        Initialize the streaming dataset.
        
        Args:
            parquet_paths: List of Parquet shard file paths
            window_size: Number of frames per window
            window_stride: Stride between windows (1 = max overlap)
            shuffle_buffer_size: Size of shuffle buffer for randomization
            frame_transform: Optional transform for each frame (PIL Image -> PIL Image)
            action_dim: Expected action dimension (for validation)
            use_gpu_decode: Whether to use GPU for JPEG decoding
            seed: Random seed for reproducibility
            shuffle_shards: Whether to shuffle shard order per epoch
            estimate_length: Whether to scan shards to estimate total windows
                             (set False to skip slow init on large datasets)
        """
        super().__init__()
        
        self.parquet_paths = sorted(parquet_paths)
        self.window_size = window_size
        self.window_stride = window_stride
        self.shuffle_buffer_size = shuffle_buffer_size
        self.frame_transform = frame_transform
        self.action_dim = action_dim
        self.use_gpu_decode = use_gpu_decode
        self.seed = seed if seed is not None else 42
        self.shuffle_shards = shuffle_shards
        
        # Epoch counter for distributed training synchronization
        self.epoch = 0
        
        # Validate
        if not self.parquet_paths:
            raise ValueError("No Parquet files provided")
        
        # Pre-compute estimated total number of windows for __len__ / tqdm
        self._estimated_length = self._compute_total_windows() if estimate_length else 0
        
        # Get distributed info for logging
        rank, world_size = self._get_distributed_info()
        
        print(f"ParquetStreamingDataset initialized (rank {rank}/{world_size}):")
        print(f"  Total shards: {len(self.parquet_paths)}")
        print(f"  Window size: {window_size}")
        print(f"  Window stride: {window_stride}")
        print(f"  Shuffle buffer: {shuffle_buffer_size}")
        print(f"  Estimated total windows: {self._estimated_length}")
        if world_size > 1:
            print(f"  Estimated windows per rank: ~{self._estimated_length // world_size}")
    
    def _compute_total_windows(self) -> int:
        """
        Estimate total number of windows by scanning parquet metadata.
        
        Reads only the episode_id column and uses pyarrow.compute.value_counts
        to aggregate per-episode frame counts without per-element Python conversion.
        """
        from collections import Counter
        
        episode_frame_counts: Counter = Counter()
        
        for path in self.parquet_paths:
            try:
                pf = pq.ParquetFile(path, memory_map=True)
                for rg_idx in range(pf.metadata.num_row_groups):
                    col = pf.read_row_group(rg_idx, columns=["episode_id"]).column("episode_id")
                    vc = pc.value_counts(col)
                    values = vc.field("values").to_pylist()
                    counts = vc.field("counts").to_pylist()
                    for v, c in zip(values, counts):
                        episode_frame_counts[v] += c
            except Exception as e:
                print(f"Warning: Could not scan {path} for length estimation: {e}")
                continue
        
        total_windows = 0
        for num_frames in episode_frame_counts.values():
            if num_frames >= self.window_size:
                total_windows += (num_frames - self.window_size) // self.window_stride + 1
        
        return total_windows
    
    def __len__(self) -> int:
        """
        Return the estimated number of windows for this rank (for tqdm progress bars).
        
        In distributed training, returns the per-rank estimate.
        """
        rank, world_size = self._get_distributed_info()
        if world_size > 1:
            return self._estimated_length // world_size
        return self._estimated_length
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this dataset.
        
        This ensures that shuffling is different across epochs while being
        consistent across all distributed processes within the same epoch.
        
        Args:
            epoch: The current epoch number
        
        Example:
            for epoch in range(num_epochs):
                dataset.set_epoch(epoch)
                for batch in dataloader:
                    train_step(batch)
        """
        self.epoch = epoch
    
    def _get_distributed_info(self) -> Tuple[int, int]:
        """
        Get the current process rank and world size for distributed training.
        
        Returns:
            Tuple of (rank, world_size). Returns (0, 1) for non-distributed training.
        """
        # First, try torch.distributed if available and initialized
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        
        # Fall back to environment variables (set by accelerate/torchrun/etc.)
        # Check multiple common environment variable names
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        return rank, world_size
    
    def _partition_shards(self, shards: List[str], partition_id: int, num_partitions: int) -> List[str]:
        """
        Partition shards evenly among partitions using interleaved assignment.
        
        Interleaved assignment (round-robin) provides better load balancing than
        contiguous assignment when shards have varying sizes.
        
        Args:
            shards: List of shard paths to partition
            partition_id: ID of this partition (0-indexed)
            num_partitions: Total number of partitions
        
        Returns:
            List of shard paths assigned to this partition
        """
        if num_partitions <= 1:
            return shards
        
        # Interleaved assignment: shard i goes to partition (i % num_partitions)
        return [s for i, s in enumerate(shards) if i % num_partitions == partition_id]
    
    def _get_rank_shards(self) -> List[str]:
        """
        Get shards assigned to this distributed rank.
        
        First shuffles all shards with a seed based on base_seed + epoch,
        then partitions among ranks.
        
        Returns:
            List of shard paths for this rank
        """
        shards = self.parquet_paths.copy()
        
        # Shuffle with epoch-dependent seed for different order each epoch
        # All ranks use the same seed so they get the same shuffle order
        if self.shuffle_shards:
            epoch_seed = self.seed + self.epoch
            rng = random.Random(epoch_seed)
            rng.shuffle(shards)
        
        # Partition among distributed ranks
        rank, world_size = self._get_distributed_info()
        return self._partition_shards(shards, rank, world_size)
    
    def _get_worker_shards(self) -> List[str]:
        """
        Get shards assigned to this DataLoader worker.
        
        This implements two-level partitioning:
        1. First, shards are partitioned among distributed ranks (_get_rank_shards)
        2. Then, each rank's shards are partitioned among DataLoader workers
        
        Returns:
            List of shard paths for this specific worker
        """
        # First level: get shards for this rank
        rank_shards = self._get_rank_shards()
        
        if not rank_shards:
            return []
        
        # Second level: partition among DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single worker: use all rank shards
            return rank_shards
        
        # Multi-worker: partition this rank's shards among workers
        return self._partition_shards(rank_shards, worker_info.id, worker_info.num_workers)
    
    _STREAM_BATCH_SIZE = 256
    _STREAM_COLUMNS = ["episode_id", "task_name", "instruction", "frame_idx", "frame_data", "action"]

    def _stream_rows_from_shards(self, shard_paths: List[str]) -> Iterator[Dict[str, Any]]:
        """Stream rows from Parquet shards in small batches to bound memory.

        Uses column projection to only read needed columns and a moderate
        batch size to limit per-batch memory from binary frame data.
        """
        for shard_path in shard_paths:
            try:
                pf = pq.ParquetFile(shard_path, memory_map=True)

                for batch in pf.iter_batches(
                    batch_size=self._STREAM_BATCH_SIZE,
                    columns=self._STREAM_COLUMNS,
                ):
                    episode_ids = batch.column("episode_id").to_pylist()
                    task_names = batch.column("task_name").to_pylist()
                    instructions = batch.column("instruction").to_pylist()
                    frame_indices = batch.column("frame_idx").to_pylist()
                    frame_datas = batch.column("frame_data").to_pylist()
                    actions = batch.column("action").to_pylist()

                    for i in range(batch.num_rows):
                        yield {
                            "episode_id": episode_ids[i],
                            "task_name": task_names[i],
                            "instruction": instructions[i],
                            "frame_idx": frame_indices[i],
                            "frame_data": frame_datas[i],
                            "action": actions[i],
                        }
                    del episode_ids, task_names, instructions, frame_indices, frame_datas, actions
            except Exception as e:
                print(f"Warning: Error reading {shard_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _process_window(
        self,
        window: Dict[str, Any],
        decoder: Optional[FrameDecoder] = None,
    ) -> Dict[str, Any]:
        """
        Process a window: decode frames, apply transforms, convert actions.
        
        Args:
            window: Raw window from SlidingWindowIterator
            decoder: Optional FrameDecoder instance for reuse (avoids repeated instantiation)
        
        Returns:
            Processed window ready for training
        """
        if decoder is None:
            decoder = FrameDecoder(use_gpu=self.use_gpu_decode)
        frames = decoder.decode_batch(window["frames_data"])
        
        # Apply transform if provided
        if self.frame_transform is not None:
            frames = [self.frame_transform(f) for f in frames]
        
        # Convert actions to tensor
        actions = torch.tensor(window["actions"], dtype=torch.float32)
        
        # Validate action dimension
        if self.action_dim is not None:
            if actions.shape[1] != self.action_dim:
                raise ValueError(
                    f"Action dim mismatch: expected {self.action_dim}, "
                    f"got {actions.shape[1]}"
                )
        
        return {
            "frames": frames,  # List[PIL.Image] - compatible with existing pipeline
            "video": frames,   # Alias for compatibility
            "actions": actions,
            "action_seq": actions,  # Alias for compatibility
            "instruction": window["instruction"],
            "prompt": window["instruction"],  # Alias for compatibility
            "task_name": window["task_name"],
            "episode_id": window["episode_id"],
        }
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the dataset, yielding processed windows.
        
        Yields:
            Processed window dicts ready for training
        """
        # Get shards for this worker
        shard_paths = self._get_worker_shards()
        rank, world_size = self._get_distributed_info()
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        
        if not shard_paths:
            print(f"[rank {rank} worker {worker_id}] WARNING: no shards assigned, yielding nothing")
            return
        
        print(
            f"[rank {rank} worker {worker_id}] epoch {self.epoch}: "
            f"{len(shard_paths)} shards assigned"
        )
        
        worker_seed = self.seed + self.epoch * 10000 + rank * 1000 + worker_id
        
        window_iter = SlidingWindowIterator(
            window_size=self.window_size,
            stride=self.window_stride,
        )
        shuffle_buffer = ShuffleBuffer(
            buffer_size=self.shuffle_buffer_size,
            seed=worker_seed,
        )
        
        decoder = FrameDecoder(use_gpu=self.use_gpu_decode)
        
        emitted_count = 0
        rows_read = 0
        windows_generated = 0
        iter_start = time.time()
        last_log_time = iter_start

        for row in self._stream_rows_from_shards(shard_paths):
            rows_read += 1
            window = window_iter.add_frame(
                frame_data=row["frame_data"],
                action=row["action"],
                frame_idx=row["frame_idx"],
                episode_id=row["episode_id"],
                instruction=row["instruction"],
                task_name=row["task_name"],
            )
            
            if window is not None:
                windows_generated += 1
                output = shuffle_buffer.add_and_sample(window)
                if output is not None:
                    emitted_count += 1
                    yield self._process_window(output, decoder=decoder)
            
            # Periodic progress log (every 60s) to help diagnose stalls
            now = time.time()
            if now - last_log_time > 60.0:
                elapsed = now - iter_start
                print(
                    f"[rank {rank} worker {worker_id}] "
                    f"rows={rows_read} windows={windows_generated} "
                    f"emitted={emitted_count} elapsed={elapsed:.0f}s "
                    f"rate={emitted_count / max(elapsed, 1):.1f} samples/s"
                )
                last_log_time = now
        
        # Flush remaining window at end
        final_window = window_iter.flush()
        if final_window is not None:
            shuffle_buffer.add_and_sample(final_window)
        
        # Flush shuffle buffer — decode on yield
        for item in shuffle_buffer.flush():
            emitted_count += 1
            yield self._process_window(item, decoder=decoder)
        
        elapsed = time.time() - iter_start
        print(
            f"[rank {rank} worker {worker_id}] epoch {self.epoch} done: "
            f"rows={rows_read} windows={windows_generated} "
            f"emitted={emitted_count} elapsed={elapsed:.0f}s"
        )


def collate_robot_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching robot trajectory samples.
    
    Args:
        batch: List of sample dicts from ParquetStreamingDataset
    
    Returns:
        Batched dict with:
            - video: List of frame lists (for existing pipeline compatibility)
            - action_seq: Tensor (B, T, D)
            - prompt: List of instruction strings
    """
    return {
        "video": [sample["video"] for sample in batch],
        "action_seq": torch.stack([sample["action_seq"] for sample in batch]),
        "prompt": [sample["prompt"] for sample in batch],
    }


def get_parquet_dataloader(
    parquet_dir: str,
    window_size: int,
    window_stride: int = 1,
    shuffle_buffer_size: int = 1000,
    batch_size: int = 1,
    num_workers: int = 0,
    frame_transform: Optional[Callable] = None,
    action_dim: Optional[int] = None,
    seed: Optional[int] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Convenience function to create a DataLoader from Parquet directory.
    
    Args:
        parquet_dir: Directory containing Parquet shards
        window_size: Frames per window
        window_stride: Stride between windows
        shuffle_buffer_size: Size of shuffle buffer
        batch_size: Batch size
        num_workers: Number of data loading workers
        frame_transform: Transform for frames
        action_dim: Expected action dimension
        seed: Random seed
        pin_memory: Pin memory for GPU transfer
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
    
    Returns:
        Configured DataLoader
    """
    import glob
    import os
    
    # Find parquet files
    parquet_paths = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    
    if not parquet_paths:
        raise ValueError(f"No Parquet files found in {parquet_dir}")
    
    # Create dataset
    dataset = ParquetStreamingDataset(
        parquet_paths=parquet_paths,
        window_size=window_size,
        window_stride=window_stride,
        shuffle_buffer_size=shuffle_buffer_size,
        frame_transform=frame_transform,
        action_dim=action_dim,
        seed=seed,
    )
    
    # Create dataloader
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_robot_batch,
        "pin_memory": pin_memory,
    }
    
    # Add prefetch_factor only if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["persistent_workers"] = persistent_workers
    
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)