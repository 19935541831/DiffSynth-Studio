import os, torch, time, threading
from queue import Queue
from accelerate import Accelerator

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, 
                 enable_wandb=True, log_interval=10, wandb_project=None, wandb_run_name=None, wandb_config=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        
        # WandB setup
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.log_interval = log_interval
        self.wandb_initialized = False
        
        if self.enable_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb is not installed. Install with: pip install wandb")
            else:
                wandb.init(
                    project=wandb_project or "diffsynth-training",
                    name=wandb_run_name,
                    config=wandb_config or {},
                    dir=output_path,
                    resume="allow",
                )
                self.wandb_initialized = True
        
        # Training speed tracking
        self.last_log_time = time.time()
        self.step_start_time = time.time()
        self.num_samples_since_last_log = 0
        
        # Async logging setup
        self._log_queue = Queue()
        self._log_thread = None
        if self.wandb_initialized:
            self._log_thread = threading.Thread(target=self._async_log_worker, daemon=True)
            self._log_thread.start()

    def _async_log_worker(self):
        """Background thread for async wandb logging."""
        while True:
            item = self._log_queue.get()
            if item is None:
                break
            try:
                metrics, step, loss_tensor = item
                if loss_tensor is not None:
                    metrics['train/loss'] = loss_tensor.item()
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"Async logging error: {e}")
            finally:
                self._log_queue.task_done()

    def _get_gpu_memory(self):
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return allocated, reserved
        return 0.0, 0.0
    
    def _compute_training_speed(self, batch_size=1):
        """Compute training speed in samples per second."""
        current_time = time.time()
        self.num_samples_since_last_log += batch_size
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
    
    def _compute_grad_norm(self, model: torch.nn.Module, norm_type: float = 2.0):
        """Compute gradient norm for all trainable parameters."""
        grad_norms = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_norms.append(param.grad.detach().norm(norm_type))
        if not grad_norms:
            return None
        total_norm = torch.norm(torch.stack(grad_norms), norm_type)
        return total_norm.item()

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        batch_size = kwargs.get('batch_size', 1)
        
        # Async log metrics to WandB (non-blocking)
        if self.wandb_initialized and accelerator.is_main_process and self.num_steps % self.log_interval == 0:
            metrics = {"step": self.num_steps}
            
            # Get loss tensor for async processing (no .item() call here)
            loss = kwargs.get('loss', None)
            loss_tensor = None
            if loss is not None and isinstance(loss, torch.Tensor):
                loss_tensor = loss.detach()
            
            # Learning rate
            optimizer = kwargs.get('optimizer', None)
            if optimizer is not None:
                metrics['train/learning_rate'] = optimizer.param_groups[0]['lr']
                for param_group in optimizer.param_groups:
                    if param_group.get('group_name', '') == 'action_encoder':
                        metrics['train/action_encoder_learning_rate'] = param_group['lr']
                        break
            
            # Gradient norm — use explicitly passed post-clip value if available
            grad_norm = kwargs.get('grad_norm', None)
            if grad_norm is not None:
                metrics['train/grad_norm'] = grad_norm
            
            # GPU memory (fast operation)
            gpu_allocated, gpu_reserved = self._get_gpu_memory()
            metrics['system/gpu_memory_allocated_gb'] = gpu_allocated
            metrics['system/gpu_memory_reserved_gb'] = gpu_reserved
            
            # Training speed
            samples_per_sec = self._compute_training_speed(batch_size)
            metrics['performance/samples_per_second'] = samples_per_sec
            
            # Queue for async logging (non-blocking)
            self._log_queue.put((metrics, self.num_steps, loss_tensor))
            
            self._reset_speed_tracking()
        
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


    def on_validation_end(self, accelerator: Accelerator, val_loss: float, step: int = None):
        if not self.wandb_initialized or (not accelerator.is_main_process):
            return
        if step is None:
            step = self.num_steps
        try:
            wandb.log({"val/loss": float(val_loss)}, step=step)
        except Exception as e:
            print(f"Validation logging error: {e}")


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
        """Close the WandB run and async logging thread."""
        if self._log_thread is not None:
            self._log_queue.put(None)
            self._log_thread.join(timeout=30)
        if self.wandb_initialized:
            wandb.finish()