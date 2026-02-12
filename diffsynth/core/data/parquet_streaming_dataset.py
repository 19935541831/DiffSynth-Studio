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
import random
import torch
import numpy as np
from PIL import Image
from collections import deque
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple
import pyarrow.parquet as pq


class SlidingWindowIterator:
    """
    Generate sliding windows from streaming trajectory data.
    
    Handles episode boundaries by clearing the buffer when a new episode starts.
    Emits windows according to the specified stride.
    
    Example:
        iterator = SlidingWindowIterator(window_size=17, stride=1)
        for frame_data, action, episode_id in stream:
            window = iterator.add_frame(frame_data, action, episode_id, instruction, task_name)
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
        episode_id: str,
        instruction: str,
        task_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Add a frame to the buffer and potentially emit a window.
        
        Args:
            frame_data: JPEG-compressed frame bytes
            action: Action vector as list of floats
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
        self.buffer.append((frame_data, action))
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
        
        return {
            "frames_data": frames_data,  # List of JPEG bytes
            "actions": actions,           # List of action vectors
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
        - CPU decoding via PIL
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
        buffer = io.BytesIO(jpeg_bytes)
        image = Image.open(buffer)
        return image.convert("RGB")
    
    def decode_batch(self, jpeg_bytes_list: List[bytes]) -> List[Image.Image]:
        """
        Decode a batch of JPEG bytes to PIL Images.
        
        Args:
            jpeg_bytes_list: List of JPEG-compressed bytes
        
        Returns:
            List of PIL Images in RGB format
        """
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
        - Optional frame transformation
    
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
        """
        super().__init__()
        
        self.parquet_paths = sorted(parquet_paths)
        self.window_size = window_size
        self.window_stride = window_stride
        self.shuffle_buffer_size = shuffle_buffer_size
        self.frame_transform = frame_transform
        self.action_dim = action_dim
        self.use_gpu_decode = use_gpu_decode
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        
        # Validate
        if not self.parquet_paths:
            raise ValueError("No Parquet files provided")
        
        # Pre-compute estimated total number of windows for __len__ / tqdm
        self._estimated_length = self._compute_total_windows()
        
        print(f"ParquetStreamingDataset initialized:")
        print(f"  Shards: {len(self.parquet_paths)}")
        print(f"  Window size: {window_size}")
        print(f"  Window stride: {window_stride}")
        print(f"  Shuffle buffer: {shuffle_buffer_size}")
        print(f"  Estimated windows: {self._estimated_length}")
    
    def _compute_total_windows(self) -> int:
        """
        Estimate total number of windows by scanning parquet metadata.
        
        Reads only the episode_id column to count frames per episode,
        then computes how many sliding windows each episode produces.
        """
        from collections import Counter
        
        episode_frame_counts: Counter = Counter()
        
        for path in self.parquet_paths:
            try:
                pf = pq.ParquetFile(path, memory_map=True)
                for rg_idx in range(pf.metadata.num_row_groups):
                    # Only read the episode_id column for efficiency
                    table = pf.read_row_group(rg_idx, columns=["episode_id"])
                    for eid in table.column("episode_id"):
                        episode_frame_counts[eid.as_py()] += 1
            except Exception as e:
                print(f"Warning: Could not scan {path} for length estimation: {e}")
                continue
        
        total_windows = 0
        for num_frames in episode_frame_counts.values():
            if num_frames >= self.window_size:
                total_windows += (num_frames - self.window_size) // self.window_stride + 1
        
        return total_windows
    
    def __len__(self) -> int:
        """Return the estimated total number of windows (for tqdm progress bars)."""
        return self._estimated_length
    
    def _get_worker_shards(self) -> List[str]:
        """Get the shard paths assigned to this worker."""
        worker_info = torch.utils.data.get_worker_info()
        
        shards = self.parquet_paths.copy()
        
        # Shuffle shards for this epoch
        if self.shuffle_shards:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(shards)
        
        if worker_info is None:
            # Single worker: use all shards
            return shards
        
        # Multi-worker: partition shards
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        
        # Distribute shards evenly
        per_worker = len(shards) // num_workers
        remainder = len(shards) % num_workers
        
        # Calculate start and end indices for this worker
        start_idx = worker_id * per_worker + min(worker_id, remainder)
        end_idx = start_idx + per_worker + (1 if worker_id < remainder else 0)
        
        return shards[start_idx:end_idx]
    
    def _stream_rows_from_shards(self, shard_paths: List[str]) -> Iterator[Dict[str, Any]]:
        """Stream rows from Parquet shards."""
        for shard_path in shard_paths:
            try:
                pf = pq.ParquetFile(shard_path, memory_map=True)
                
                # Read row groups for streaming
                for rg_idx in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(rg_idx)
                    
                    for i in range(table.num_rows):
                        yield {
                            "episode_id": table.column("episode_id")[i].as_py(),
                            "task_name": table.column("task_name")[i].as_py(),
                            "instruction": table.column("instruction")[i].as_py(),
                            "frame_idx": table.column("frame_idx")[i].as_py(),
                            "frame_data": table.column("frame_data")[i].as_py(),
                            "action": table.column("action")[i].as_py(),
                        }
            except Exception as e:
                print(f"Warning: Error reading {shard_path}: {e}")
                continue
    
    def _process_window(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a window: decode frames, apply transforms, convert actions.
        
        Args:
            window: Raw window from SlidingWindowIterator
        
        Returns:
            Processed window ready for training
        """
        # Decode frames
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
        
        if not shard_paths:
            return
        
        # Initialize components
        window_iter = SlidingWindowIterator(
            window_size=self.window_size,
            stride=self.window_stride,
        )
        shuffle_buffer = ShuffleBuffer(
            buffer_size=self.shuffle_buffer_size,
            seed=self.seed,
        )
        
        # Stream through shards
        for row in self._stream_rows_from_shards(shard_paths):
            # Generate window from frame
            window = window_iter.add_frame(
                frame_data=row["frame_data"],
                action=row["action"],
                episode_id=row["episode_id"],
                instruction=row["instruction"],
                task_name=row["task_name"],
            )
            
            if window is not None:
                # Process and add to shuffle buffer
                processed = self._process_window(window)
                output = shuffle_buffer.add_and_sample(processed)
                
                if output is not None:
                    yield output
        
        # Flush remaining window at end
        final_window = window_iter.flush()
        if final_window is not None:
            processed = self._process_window(final_window)
            shuffle_buffer.add_and_sample(processed)
        
        # Flush shuffle buffer
        for item in shuffle_buffer.flush():
            yield item


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
