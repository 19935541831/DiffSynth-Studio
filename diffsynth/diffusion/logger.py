import os, torch, time
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, 
                 enable_tensorboard=True, log_interval=10):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        
        # TensorBoard setup
        self.enable_tensorboard = enable_tensorboard
        self.log_interval = log_interval
        if enable_tensorboard:
            log_dir = os.path.join(output_path, "logs")
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        
        # Training speed tracking
        self.last_log_time = time.time()
        self.step_start_time = time.time()
        self.num_samples_since_last_log = 0


    def _compute_grad_norm(self, model: torch.nn.Module):
        """Compute the L2 norm of gradients for monitoring gradient explosion."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _get_gpu_memory(self):
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            return allocated, reserved
        return 0.0, 0.0
    
    def _compute_training_speed(self, batch_size=1):
        """Compute training speed in samples per second."""
        current_time = time.time()
        
        # Accumulate samples processed since last log
        self.num_samples_since_last_log += batch_size
        
        # Compute speed over the interval
        time_elapsed = current_time - self.last_log_time
        
        if time_elapsed > 0:
            samples_per_sec = self.num_samples_since_last_log / time_elapsed
        else:
            samples_per_sec = 0.0
        
        return samples_per_sec
    
    def _reset_speed_tracking(self):
        """Reset speed tracking after logging."""
        self.last_log_time = time.time()
        self.num_samples_since_last_log = 0

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        
        # Accumulate samples for speed calculation (do this every step)
        batch_size = kwargs.get('batch_size', 1)
        
        # Log metrics to TensorBoard
        if self.writer is not None and accelerator.is_main_process and self.num_steps % self.log_interval == 0:
            # Log loss
            loss = kwargs.get('loss', None)
            if loss is not None:
                loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
                self.writer.add_scalar('train/loss', loss_value, self.num_steps)
            
            # Log learning rate
            optimizer = kwargs.get('optimizer', None)
            if optimizer is not None:
                current_lr = optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/learning_rate', current_lr, self.num_steps)
            
            # Log gradient norm
            grad_norm = self._compute_grad_norm(model)
            self.writer.add_scalar('train/grad_norm', grad_norm, self.num_steps)
            
            # Log GPU memory
            gpu_allocated, gpu_reserved = self._get_gpu_memory()
            self.writer.add_scalar('system/gpu_memory_allocated_gb', gpu_allocated, self.num_steps)
            self.writer.add_scalar('system/gpu_memory_reserved_gb', gpu_reserved, self.num_steps)
            
            # Log training speed (compute based on accumulated samples)
            samples_per_sec = self._compute_training_speed(batch_size)
            self.writer.add_scalar('performance/samples_per_second', samples_per_sec, self.num_steps)
            
            # Reset speed tracking after logging
            self._reset_speed_tracking()
            
            # Flush writer
            self.writer.flush()
        
        # Save checkpoint
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
