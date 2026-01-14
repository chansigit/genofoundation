"""
VAE Training Script
支持：
- 可配置的训练参数
- 学习率调度
- Early stopping
- 检查点保存与恢复
- TensorBoard / WandB 日志
- 验证集评估
"""

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    ReduceLROnPlateau, 
    OneCycleLR,
    LinearLR,
    SequentialLR,
)

# 假设 VAE 类在同一目录或已导入
# from vae import VAE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本参数
    epochs: int = 100                    # Total epochs for complete training (used for absolute progress)
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # VAE 特定参数
    beta: float = 1.0                    # KL 权重
    beta_warmup_epochs: int = 10         # Beta 预热轮数
    beta_schedule: str = "linear"        # "linear", "cyclical", "constant"
    
    # 优化器
    optimizer: str = "adamw"             # "adam", "adamw", "sgd"
    betas: tuple = (0.9, 0.999)
    
    # 学习率调度
    scheduler: str = "cosine"            # "cosine", "plateau", "onecycle", "warmup_cosine", "warmup"
    lr_warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    
    # 检查点
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10                 # 每 N 轮保存
    save_best: bool = True
    
    # 日志
    log_every: int = 100                 # 每 N 步打印
    use_wandb: bool = False
    use_tensorboard: bool = True
    project_name: str = "vae_training"
    run_name: Optional[str] = None
    
    # 设备
    device: str = "cuda"
    mixed_precision: bool = True         # 使用 AMP
    
    # 数据
    num_workers: int = 4
    pin_memory: bool = True


# ============================================================================
# VAE's beta scheduler
# ============================================================================
class BetaScheduler:
    """KL weight scheduler for VAE"""
    def __init__(
        self, 
        schedule: str = "linear",
        initial_beta: float = 0.0,
        final_beta: float = 1.0,
        warmup_epochs: int = 10,
        cycle_epochs: int = 10,  # for cyclical
    ):
        self.schedule = schedule
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.warmup_epochs = warmup_epochs
        self.cycle_epochs = cycle_epochs
        self.current_beta = initial_beta
    
    def step(self, epoch: int) -> float:
        if self.schedule == "constant":
            self.current_beta = self.final_beta
            
        elif self.schedule == "linear":
            if epoch < self.warmup_epochs:
                progress = epoch / max(1, self.warmup_epochs)
                self.current_beta = self.initial_beta + progress * (self.final_beta - self.initial_beta)
            else:
                self.current_beta = self.final_beta
                
        elif self.schedule == "cyclical":
            # Cyclical annealing: https://arxiv.org/abs/1903.10145
            cycle = epoch // self.cycle_epochs
            position = (epoch % self.cycle_epochs) / max(1, self.cycle_epochs)
            self.current_beta = min(self.final_beta, position * self.final_beta)
            
        elif self.schedule == "sigmoid":
            if epoch < self.warmup_epochs:
                x = (epoch - self.warmup_epochs / 2) / (self.warmup_epochs / 4)
                self.current_beta = self.final_beta / (1 + math.exp(-x))
            else:
                self.current_beta = self.final_beta
        
        return self.current_beta


# ============================================================================
# Early Stopping
# ============================================================================
class EarlyStopping:
    """Early stopping handler. 
    Training continues for up to {patience} epochs without seeing improvement before stopping.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


# ============================================================================
# Trainer
# ============================================================================

class VAETrainer:
    """VAE 训练器"""
    
    def __init__(
        self,
        model: nn.Module,  # VAE model
        config: TrainingConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        
        # 支持 dict 或 TrainingConfig
        if isinstance(config, dict):
            self.config = TrainingConfig(**config)
        else:
            self.config = config
            
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
            )
        else:
            self.val_loader = None
        
        # 训练状态 - Absolute progress tracking (set early for scheduler creation)
        self.total_epochs_planned = config.epochs      # Total epochs for complete training
        self.absolute_epoch = 0                        # True position in full training plan
        self.absolute_step = 0                         # Global step in full training plan
        
        # Legacy/compatibility fields (derived from absolute progress)
        self.current_epoch = 0                         # Current training epoch (for compatibility)
        self.global_step = 0                          # Current global step (for compatibility)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器 (requires total_epochs_planned)
        self.scheduler = self._create_scheduler()
        
        # Beta 调度器
        self.beta_scheduler = BetaScheduler(
            schedule=config.beta_schedule,
            initial_beta=0.0,
            final_beta=config.beta,
            warmup_epochs=config.beta_warmup_epochs,
        )
        
        # Early stopping
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta,
            )
        else:
            self.early_stopping = None
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
        
        # 日志
        self.writer = None
        self.wandb_run = None
        self._setup_logging()
        
        # 检查点目录
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        params = self.model.parameters()
        
        if self.config.optimizer == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """创建学习率调度器 - 基于完整训练计划，不考虑resume位置"""
        # Always use full training plan for scheduler creation
        total_steps = len(self.train_loader) * self.total_epochs_planned
        warmup_steps = len(self.train_loader) * self.config.lr_warmup_epochs
        
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_epochs_planned,  # Use full training plan
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr,
            )
        elif self.config.scheduler == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
            )
        elif self.config.scheduler == "warmup_cosine":
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_epochs_planned - self.config.lr_warmup_epochs,  # Use full training plan
                eta_min=self.config.min_lr,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.config.lr_warmup_epochs],
            )
        elif self.config.scheduler == "warmup":
            # Pure warmup stage - lr increases linearly then stays constant
            return LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.config.lr_warmup_epochs,
            )
        else:
            return None
    
    def _setup_logging(self):
        """设置日志"""
        if self.config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(self.config.checkpoint_dir, "logs")
            self.writer = SummaryWriter(log_dir)
        
        if self.config.use_wandb:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=vars(self.config),
            )
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """记录指标"""
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{name}", value, step)
        
        if self.wandb_run is not None:
            import wandb
            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        current_beta = self.beta_scheduler.step(self.absolute_epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 处理不同格式的 batch
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
                condition = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                x = batch.to(self.device)
                condition = None
            
            self.optimizer.zero_grad()
            
            # 前向传播（支持混合精度）
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(x, condition=condition)
                    losses = self.model.loss(x, outputs, beta=current_beta)
                    loss = losses['total']
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x, condition=condition)
                losses = self.model.loss(x, outputs, beta=current_beta)
                loss = losses['total']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_recon_loss += losses['reconstruction'].item()
            total_kl_loss += losses['kl'].item()
            num_batches += 1
            self.global_step += 1
            
            # 日志
            if batch_idx % self.config.log_every == 0:
                logger.info(
                    f"Epoch [{self.current_epoch+1}/{self.config.epochs}] "
                    f"Step [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Recon: {losses['reconstruction'].item():.4f} "
                    f"KL: {losses['kl'].item():.4f} "
                    f"Beta: {current_beta:.4f}"
                )
            
            # OneCycle scheduler 每步更新
            if self.config.scheduler == "onecycle" and self.scheduler is not None:
                self.scheduler.step()
        
        # 计算平均损失
        avg_metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'beta': current_beta,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
                condition = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                x = batch.to(self.device)
                condition = None
            
            outputs = self.model(x, condition)
            losses = self.model.loss(x, outputs, beta=self.config.beta)
            
            total_loss += losses['total'].item()
            total_recon_loss += losses['reconstruction'].item()
            total_kl_loss += losses['kl'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
        }
    
    def _get_beta_scheduler_state(self) -> dict:
        """Get beta scheduler state for checkpointing"""
        return {
            'current_beta': self.beta_scheduler.current_beta,
            'schedule': self.beta_scheduler.schedule,
            'initial_beta': self.beta_scheduler.initial_beta,
            'final_beta': self.beta_scheduler.final_beta,
            'warmup_epochs': self.beta_scheduler.warmup_epochs,
            'cycle_epochs': self.beta_scheduler.cycle_epochs,
        }
    
    def _get_early_stopping_state(self) -> Optional[dict]:
        """Get early stopping state for checkpointing"""
        if not self.early_stopping:
            return None
        return {
            'counter': self.early_stopping.counter,
            'best_score': self.early_stopping.best_score,
            'should_stop': self.early_stopping.should_stop,
        }
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            # Enhanced progress tracking
            'absolute_epoch': self.absolute_epoch,
            'absolute_step': self.absolute_step,
            'total_epochs_planned': self.total_epochs_planned,
            
            # Legacy/compatibility fields
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            
            # Model and optimizer states
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            # Enhanced scheduler states
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'beta_scheduler_state': self._get_beta_scheduler_state(),
            'early_stopping_state': self._get_early_stopping_state(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            
            # Training metrics and metadata
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def _restore_beta_scheduler_state(self, state: Optional[dict]):
        """Restore beta scheduler state from checkpoint"""
        if not state:
            return
        
        # Verify structural parameters match (for safety)
        if (state.get('schedule') != self.beta_scheduler.schedule or
            state.get('initial_beta') != self.beta_scheduler.initial_beta or
            state.get('final_beta') != self.beta_scheduler.final_beta):
            logger.warning("Beta scheduler config mismatch, recalculating from absolute epoch")
        
        # Recalculate current_beta based on absolute_epoch for consistency
        # This ensures resumed training follows the same beta curve as uninterrupted training
        self.beta_scheduler.current_beta = self.beta_scheduler.step(self.absolute_epoch)
        logger.info(f"Beta scheduler restored: current_beta={self.beta_scheduler.current_beta:.4f} at epoch {self.absolute_epoch}")
    
    def _restore_early_stopping_state(self, state: Optional[dict]):
        """Restore early stopping state from checkpoint"""
        if not state or not self.early_stopping:
            return
        
        self.early_stopping.counter = state['counter']
        self.early_stopping.best_score = state['best_score']
        self.early_stopping.should_stop = state['should_stop']
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore absolute progress tracking
        self.absolute_epoch = checkpoint.get('absolute_epoch', checkpoint.get('epoch', 0))
        self.absolute_step = checkpoint.get('absolute_step', checkpoint.get('global_step', 0))
        self.total_epochs_planned = checkpoint.get('total_epochs_planned', self.config.epochs)
        
        # Set legacy/compatibility fields
        self.current_epoch = self.absolute_epoch
        self.global_step = self.absolute_step
        
        # Restore training metrics
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        # Store scheduler state for later restoration (after fast-forward)
        self._saved_scheduler_state = checkpoint.get('scheduler_state_dict')
        # Note: We don't restore scheduler state here because scheduler needs to be 
        # fast-forwarded first, then state can be restored
        
        self._restore_beta_scheduler_state(checkpoint.get('beta_scheduler_state'))
        self._restore_early_stopping_state(checkpoint.get('early_stopping_state'))
        
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from absolute epoch {self.absolute_epoch}")
        logger.info(f"Total epochs planned: {self.total_epochs_planned}")
    
    def _fast_forward_scheduler(self, target_epoch: int):
        """Fast-forward scheduler to target epoch position"""
        if not self.scheduler or target_epoch <= 0:
            return
        
        logger.info(f"Fast-forwarding scheduler to epoch {target_epoch}")
        
        # Different strategies based on scheduler type
        if isinstance(self.scheduler, (CosineAnnealingLR, LinearLR)):
            # Epoch-based schedulers - step once per epoch
            for _ in range(target_epoch):
                self.scheduler.step()
                
        elif isinstance(self.scheduler, SequentialLR):
            # Sequential scheduler - step once per epoch
            for _ in range(target_epoch):
                self.scheduler.step()
                
        elif isinstance(self.scheduler, OneCycleLR):
            # Step-based scheduler - step once per batch
            target_steps = target_epoch * len(self.train_loader)
            for _ in range(target_steps):
                self.scheduler.step()
                
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            # Plateau scheduler is stateful - should be restored from checkpoint
            # No fast-forward needed as state is loaded directly
            pass
        else:
            # Generic fallback - assume epoch-based
            logger.warning(f"Unknown scheduler type {type(self.scheduler)}, using epoch-based fast-forward")
            for _ in range(target_epoch):
                self.scheduler.step()
    
    def train(self, resume_from: Optional[str] = None):
        """完整训练循环 - 使用绝对进度跟踪"""
        # Load checkpoint first (sets absolute_epoch)
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Fast-forward scheduler to correct position if resuming
        if resume_from and self.scheduler:
            # Try to restore scheduler state first if available
            if hasattr(self, '_saved_scheduler_state') and self._saved_scheduler_state:
                try:
                    self.scheduler.load_state_dict(self._saved_scheduler_state)
                    logger.info("Restored scheduler state from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to restore scheduler state: {e}, falling back to fast-forward")
                    self._fast_forward_scheduler(self.absolute_epoch)
            else:
                # Fall back to fast-forwarding
                self._fast_forward_scheduler(self.absolute_epoch)
        
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training from epoch {self.absolute_epoch} to {self.total_epochs_planned}")
        
        # Training loop with absolute progress
        for epoch in range(self.absolute_epoch, self.total_epochs_planned):
            self.absolute_epoch = epoch
            self.current_epoch = epoch  # For compatibility
            self.absolute_step += len(self.train_loader)  # Update absolute step
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self._log_metrics(train_metrics, epoch, prefix="train")
            
            # 验证
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['loss'])
                self._log_metrics(val_metrics, epoch, prefix="val")
                
                val_loss = val_metrics['loss']
                is_best = val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = val_loss
                
                # 学习率调度（如果基于验证损失）
                if self.config.scheduler == "plateau" and self.scheduler:
                    self.scheduler.step(val_loss)
            else:
                is_best = False
            
            # 其他调度器
            if self.scheduler and self.config.scheduler not in ["plateau", "onecycle"]:
                self.scheduler.step()
            
            # 打印 epoch 总结
            log_msg = (
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
            )
            if val_metrics:
                log_msg += (
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"Val Recon: {val_metrics['recon_loss']:.4f} "
                )
            logger.info(log_msg)
            
            # 保存检查点
            if self.config.save_best and is_best:
                self.save_checkpoint("best_model.pt", is_best=True)
            
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Early stopping
            if self.early_stopping and val_metrics:
                if self.early_stopping(val_metrics['recon_loss']):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 保存最终模型
        self.save_checkpoint("final_model.pt")
        
        # 清理
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()
        
        logger.info("Training completed!")
        return self.train_losses, self.val_losses


# ============================================================================
# 使用示例
# ============================================================================

def create_dummy_dataset(num_samples: int = 10000, input_dim: int = 784):
    """创建示例数据集"""
    class DummyDataset(Dataset):
        def __init__(self, num_samples, input_dim):
            self.data = torch.randn(num_samples, input_dim)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return DummyDataset(num_samples, input_dim)


if __name__ == "__main__":
    # 示例：训练 VAE
    
    # 配置
    config = TrainingConfig(
        epochs=50,
        batch_size=256,
        learning_rate=1e-3,
        beta=1.0,
        beta_warmup_epochs=10,
        beta_schedule="linear",
        scheduler="warmup_cosine",
        lr_warmup_epochs=5,
        early_stopping=True,
        patience=10,
        use_tensorboard=True,
        use_wandb=False,
        mixed_precision=True,
    )
    
    # 创建模型（需要导入实际的 VAE 类）
    # model = VAE(
    #     input_dim=784,
    #     latent_dim=64,
    #     encoder_hidden_dims=[512, 256],
    #     decoder_hidden_dims=[256, 512],
    #     activation="silu",
    #     dropout=0.1,
    #     use_layer_norm=True,
    #     output_activation="sigmoid",
    #     beta=1.0,
    # )
    
    # 用户需要自己准备并传入训练集和验证集
    # train_dataset = YourTrainDataset(...)
    # val_dataset = YourValDataset(...)  # 可选，传 None 则跳过验证
    
    # 创建训练器（直接传入已准备好的数据集）
    # trainer = VAETrainer(
    #     model=model,
    #     config=config,
    #     train_dataset=train_dataset,  # 必须
    #     val_dataset=val_dataset,       # 可选
    # )
    
    # 开始训练
    # train_losses, val_losses = trainer.train()
    
    # 从检查点恢复训练
    # trainer.train(resume_from="checkpoints/checkpoint_epoch_20.pt")
    
    print("Training script ready. Uncomment the code above to run.")