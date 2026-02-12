from .operators import *
import torch, json, pandas
import imageio
import warnings


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
        enable_sliding_window=False,
        window_num_frames=None,
        window_stride=1,
        video_key="video",
        action_key="action_seq",
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        
        # Sliding window parameters
        self.enable_sliding_window = enable_sliding_window
        self.window_num_frames = window_num_frames
        self.window_stride = window_stride
        self.video_key = video_key
        self.action_key = action_key
        self.window_indices = []  # [(video_idx, start_frame), ...]
        
        self.load_metadata(metadata_path)
        
        # Build sliding window indices if enabled
        if self.enable_sliding_window:
            if self.load_from_cache:
                raise ValueError("Sliding window mode is not compatible with cached data loading.")
            if self.window_num_frames is None:
                raise ValueError("window_num_frames must be specified when enable_sliding_window=True")
            self._build_sliding_windows()
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
    
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def _get_video_length(self, metadata):
        """Get video length from metadata or by reading the video file."""
        # First, try to get from metadata (preferred method)
        if "num_frames" in metadata:
            return int(metadata["num_frames"])
        
        # If not in metadata, read from video file
        video_path = metadata.get(self.video_key)
        if video_path is None:
            raise ValueError(f"Video key '{self.video_key}' not found in metadata: {metadata}")
        
        # Convert to absolute path
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.base_path, video_path)
        
        # Read video length using imageio
        try:
            reader = imageio.get_reader(video_path)
            num_frames = int(reader.count_frames())
            reader.close()
            return num_frames
        except Exception as e:
            raise IOError(f"Failed to read video length from {video_path}: {str(e)}")
    
    def _build_sliding_windows(self):
        """Build window indices for sliding window sampling."""
        print(f"Building sliding window indices (num_frames={self.window_num_frames}, stride={self.window_stride})...")
        self.window_indices = []
        
        for video_idx, metadata in enumerate(self.data):
            try:
                video_length = self._get_video_length(metadata)
            except Exception as e:
                warnings.warn(f"Skipping video {video_idx} due to error: {str(e)}")
                continue
            
            # Check if video is long enough
            if video_length < self.window_num_frames:
                video_path = metadata.get(self.video_key, "unknown")
                raise ValueError(
                    f"Video {video_idx} (path: {video_path}) has {video_length} frames, "
                    f"which is less than required window size {self.window_num_frames}"
                )
            
            # Calculate number of windows for this video
            num_windows = (video_length - self.window_num_frames) // self.window_stride + 1
            
            # Add window indices
            for i in range(num_windows):
                start_frame = i * self.window_stride
                self.window_indices.append((video_idx, start_frame))
        
        print(f"Built {len(self.window_indices)} windows from {len(self.data)} videos")
    
    def _create_windowed_operators(self, frame_range):
        """Create operators with frame_range set for sliding window mode."""
        # Create a new main_data_operator with frame_range
        windowed_main_operator = self.main_data_operator
        
        # For special operators, we need to inject frame_range
        windowed_special_map = {}
        for key, operator in self.special_operator_map.items():
            # Check if this is a video or action operator that needs frame_range
            if key == self.video_key or key == self.action_key:
                # We need to reconstruct the operator pipeline with frame_range
                # This is a bit tricky because we need to inject frame_range into LoadVideo/LoadActionSequence
                windowed_special_map[key] = operator
            else:
                windowed_special_map[key] = operator
        
        return windowed_main_operator, windowed_special_map

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        elif self.enable_sliding_window:
            # Sliding window mode
            window_idx = data_id % len(self.window_indices)
            video_idx, start_frame = self.window_indices[window_idx]
            end_frame = start_frame + self.window_num_frames
            
            # Get metadata for this video
            metadata = self.data[video_idx].copy()
            
            # Process data with frame_range
            for key in self.data_file_keys:
                if key in metadata:
                    if key in self.special_operator_map:
                        operator = self.special_operator_map[key]
                        # Inject frame_range for video and action keys
                        if key == self.video_key or key == self.action_key:
                            # We need to temporarily modify the operator to use frame_range
                            # This requires walking through the operator pipeline
                            metadata[key] = self._apply_operator_with_frame_range(
                                operator, metadata[key], (start_frame, end_frame)
                            )
                        else:
                            metadata[key] = operator(metadata[key])
                    elif key in self.data_file_keys:
                        metadata[key] = self.main_data_operator(metadata[key])
            
            data = metadata
        else:
            # Normal mode
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data
    
    def _apply_operator_with_frame_range(self, operator, data, frame_range):
        """Apply operator with frame_range by walking through the pipeline."""
        if isinstance(operator, DataProcessingPipeline):
            # Walk through the pipeline and inject frame_range into LoadVideo/LoadActionSequence
            for op in operator.operators:
                if isinstance(op, (LoadVideo, LoadGIF)):
                    # Temporarily set frame_range
                    old_frame_range = op.frame_range
                    op.frame_range = frame_range
                    result = operator(data)
                    op.frame_range = old_frame_range
                    return result
                elif isinstance(op, LoadActionSequence):
                    # Temporarily set frame_range
                    old_frame_range = op.frame_range
                    op.frame_range = frame_range
                    result = operator(data)
                    op.frame_range = old_frame_range
                    return result
            # If no LoadVideo/LoadActionSequence found, just apply normally
            return operator(data)
        elif isinstance(operator, (LoadVideo, LoadGIF, LoadActionSequence)):
            # Directly modify and apply
            old_frame_range = operator.frame_range
            operator.frame_range = frame_range
            result = operator(data)
            operator.frame_range = old_frame_range
            return result
        else:
            # For other operators, just apply normally
            return operator(data)

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        elif self.enable_sliding_window:
            return len(self.window_indices) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
