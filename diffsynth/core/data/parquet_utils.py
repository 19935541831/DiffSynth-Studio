"""
Parquet Utilities for Robot Data Storage

This module provides utilities for storing and loading robot trajectory data
in Parquet format with row-per-frame storage for efficient streaming.

Schema Design:
    - episode_id: Unique trajectory identifier
    - task_name: Task category  
    - instruction: Scene-level instruction text
    - total_frames: Total frames in the trajectory
    - frame_idx: Frame index within trajectory (0-indexed)
    - frame_data: JPEG-compressed frame bytes
    - action: Action vector as list of floats [joint_dim]
    - timestamp: Optional frame timestamp
"""

import io
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any, Iterator, Tuple
import pyarrow as pa
import pyarrow.parquet as pq


# Define the Parquet schema for robot trajectory data
ROBOT_TRAJECTORY_SCHEMA = pa.schema([
    pa.field("episode_id", pa.string()),
    pa.field("task_name", pa.string()),
    pa.field("instruction", pa.string()),
    pa.field("total_frames", pa.int32()),
    pa.field("frame_idx", pa.int32()),
    pa.field("frame_data", pa.binary()),
    pa.field("action", pa.list_(pa.float32())),
    pa.field("timestamp", pa.float64()),
])


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 95) -> bytes:
    """
    Encode a frame (numpy array or PIL Image) to JPEG bytes.
    
    Args:
        frame: Image as numpy array (H, W, C) in RGB format or PIL Image
        quality: JPEG quality (1-100)
    
    Returns:
        JPEG-compressed bytes
    """
    if isinstance(frame, np.ndarray):
        image = Image.fromarray(frame)
    else:
        image = frame
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_jpeg_to_frame(jpeg_bytes: bytes) -> Image.Image:
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


def decode_jpeg_to_numpy(jpeg_bytes: bytes) -> np.ndarray:
    """
    Decode JPEG bytes to numpy array.
    
    Args:
        jpeg_bytes: JPEG-compressed bytes
    
    Returns:
        Numpy array (H, W, C) in RGB format
    """
    image = decode_jpeg_to_frame(jpeg_bytes)
    return np.array(image)


class ParquetTrajectoryWriter:
    """
    Writer for streaming trajectory data to Parquet files with automatic sharding.
    
    Example:
        writer = ParquetTrajectoryWriter("output_dir", shard_size=10000)
        for frame, action in trajectory:
            writer.write_frame(
                episode_id="ep_001",
                task_name="pick_place",
                instruction="Pick up the red cube",
                total_frames=100,
                frame_idx=0,
                frame=frame,
                action=action
            )
        writer.close()
    """
    
    def __init__(
        self,
        output_dir: str,
        shard_size: int = 10000,
        jpeg_quality: int = 95,
        compression: str = "snappy",
    ):
        """
        Initialize the writer.
        
        Args:
            output_dir: Directory to write Parquet shards
            shard_size: Number of rows per shard
            jpeg_quality: JPEG compression quality (1-100)
            compression: Parquet compression algorithm
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.jpeg_quality = jpeg_quality
        self.compression = compression
        
        self.current_shard_idx = 0
        self.current_rows: List[Dict[str, Any]] = []
        self.total_rows_written = 0
    
    def write_frame(
        self,
        episode_id: str,
        task_name: str,
        instruction: str,
        total_frames: int,
        frame_idx: int,
        frame: np.ndarray,
        action: np.ndarray,
        timestamp: float = 0.0,
    ):
        """
        Write a single frame to the buffer, flushing to disk when shard is full.
        
        Args:
            episode_id: Unique trajectory identifier
            task_name: Task category
            instruction: Scene-level instruction text
            total_frames: Total frames in this trajectory
            frame_idx: Current frame index (0-indexed)
            frame: Frame image as numpy array (H, W, C) RGB
            action: Action vector as numpy array (joint_dim,)
            timestamp: Optional frame timestamp
        """
        # Encode frame to JPEG
        frame_bytes = encode_frame_to_jpeg(frame, self.jpeg_quality)
        
        # Create row
        row = {
            "episode_id": episode_id,
            "task_name": task_name,
            "instruction": instruction,
            "total_frames": total_frames,
            "frame_idx": frame_idx,
            "frame_data": frame_bytes,
            "action": action.astype(np.float32).tolist(),
            "timestamp": timestamp,
        }
        
        self.current_rows.append(row)
        
        # Flush if shard is full
        if len(self.current_rows) >= self.shard_size:
            self._flush_shard()
    
    def _flush_shard(self):
        """Write current buffer to a Parquet shard file."""
        if not self.current_rows:
            return
        
        import os
        
        # Convert to PyArrow table
        table = pa.Table.from_pylist(self.current_rows, schema=ROBOT_TRAJECTORY_SCHEMA)
        
        # Write to file
        shard_path = os.path.join(
            self.output_dir,
            f"shard_{self.current_shard_idx:06d}.parquet"
        )
        pq.write_table(
            table,
            shard_path,
            compression=self.compression,
        )
        
        self.total_rows_written += len(self.current_rows)
        print(f"  Written shard {self.current_shard_idx} with {len(self.current_rows)} rows")
        
        # Reset for next shard
        self.current_shard_idx += 1
        self.current_rows = []
    
    def close(self):
        """Flush any remaining data and finalize."""
        self._flush_shard()
        print(f"Total rows written: {self.total_rows_written}")
        print(f"Total shards: {self.current_shard_idx}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class ParquetTrajectoryReader:
    """
    Reader for streaming trajectory data from Parquet shards.
    
    Supports:
        - Streaming row-by-row iteration
        - Memory-mapped reading for efficiency
        - Multi-file shard reading
    
    Example:
        reader = ParquetTrajectoryReader(["shard_0.parquet", "shard_1.parquet"])
        for row in reader:
            frame = decode_jpeg_to_frame(row["frame_data"])
            action = np.array(row["action"])
    """
    
    def __init__(
        self,
        parquet_paths: List[str],
        columns: Optional[List[str]] = None,
        memory_map: bool = True,
    ):
        """
        Initialize the reader.
        
        Args:
            parquet_paths: List of Parquet file paths to read
            columns: Columns to load (None for all)
            memory_map: Whether to use memory-mapped reading
        """
        self.parquet_paths = sorted(parquet_paths)
        self.columns = columns
        self.memory_map = memory_map
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all rows in all shards."""
        for path in self.parquet_paths:
            pf = pq.ParquetFile(path, memory_map=self.memory_map)
            
            # Read row groups one at a time for streaming
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=self.columns)
                
                # Convert to Python dicts
                for i in range(table.num_rows):
                    row = {col: table.column(col)[i].as_py() for col in table.column_names}
                    yield row
    
    def iter_batches(self, batch_size: int = 1000) -> Iterator[pa.Table]:
        """
        Iterate over batches of rows as PyArrow Tables.
        
        Args:
            batch_size: Number of rows per batch
        
        Yields:
            PyArrow Table batches
        """
        for path in self.parquet_paths:
            pf = pq.ParquetFile(path, memory_map=self.memory_map)
            
            for batch in pf.iter_batches(batch_size=batch_size, columns=self.columns):
                yield pa.Table.from_batches([batch])
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get combined metadata from all shards."""
        total_rows = 0
        episode_ids = set()
        
        for path in self.parquet_paths:
            pf = pq.ParquetFile(path, memory_map=self.memory_map)
            total_rows += pf.metadata.num_rows
            
            # Sample first row group to get episode IDs
            table = pf.read_row_group(0, columns=["episode_id"])
            for i in range(table.num_rows):
                episode_ids.add(table.column("episode_id")[i].as_py())
        
        return {
            "total_rows": total_rows,
            "num_shards": len(self.parquet_paths),
            "sample_episode_ids": list(episode_ids)[:10],
        }


def get_parquet_shard_paths(data_dir: str, pattern: str = "shard_*.parquet") -> List[str]:
    """
    Get sorted list of Parquet shard paths from a directory.
    
    Args:
        data_dir: Directory containing Parquet files
        pattern: Glob pattern for shard files
    
    Returns:
        Sorted list of absolute paths
    """
    import os
    import glob
    
    paths = glob.glob(os.path.join(data_dir, pattern))
    return sorted(paths)
