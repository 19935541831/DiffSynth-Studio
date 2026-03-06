import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    validation_dataset: torch.utils.data.Dataset = None,
    validation_steps: int = None,
    validation_num_batches: int = None,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        validation_steps = getattr(args, "validation_steps", validation_steps)
        validation_num_batches = getattr(args, "validation_num_batches", validation_num_batches)
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    validation_dataloader = None
    if validation_dataset is not None:
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    if validation_dataloader is not None:
        model, optimizer, dataloader, validation_dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, validation_dataloader, scheduler)
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    def run_validation(step: int):
        if validation_dataloader is None:
            return
        model.eval()
        val_loss_sum = torch.tensor(0.0, device=accelerator.device)
        val_loss_count = torch.tensor(0, device=accelerator.device, dtype=torch.long)
        with torch.no_grad():
            for batch_id, val_data in enumerate(validation_dataloader):
                if validation_num_batches is not None and validation_num_batches > 0 and batch_id >= validation_num_batches:
                    break
                if getattr(validation_dataset, "load_from_cache", False):
                    loss = model({}, inputs=val_data)
                else:
                    loss = model(val_data)
                val_loss_sum += loss.detach().float()
                val_loss_count += 1
        if accelerator.num_processes > 1:
            accelerator.reduce(val_loss_sum, reduction="sum")
            accelerator.reduce(val_loss_count, reduction="sum")
        if val_loss_count.item() > 0:
            mean_val_loss = (val_loss_sum / val_loss_count).item()
            model_logger.on_validation_end(accelerator, mean_val_loss, step=step)
        model.train()
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, optimizer=optimizer, batch_size=1)
                if validation_dataloader is not None and validation_steps is not None and validation_steps > 0 and model_logger.num_steps % validation_steps == 0:
                    run_validation(model_logger.num_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
    model_logger.close()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
