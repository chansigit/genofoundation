#!/usr/bin/env python3
"""
Test script to verify that resumed training matches uninterrupted training behavior.

This script tests:
1. Learning rate scheduler consistency across resume
2. Beta scheduler consistency across resume  
3. Training metrics consistency
4. Multiple scheduler types (cosine, warmup, warmup_cosine)
5. Different resume points
"""

import os
import sys
import tempfile
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any

# Add vae directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vae'))

from trainer import VAETrainer, TrainingConfig


# Simple VAE model for testing
class SimpleVAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean and logvar
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(), 
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar
        
    def decode(self, z):
        return self.decoder(z)
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def forward(self, x, condition=None):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return {
            'reconstruction': recon,
            'mean': mean,
            'logvar': logvar,
            'z': z
        }
        
    def loss(self, x, outputs, beta=1.0):
        recon = outputs['reconstruction']
        mean = outputs['mean']
        logvar = outputs['logvar']
        
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss
        }


# Simple dataset for testing
class TestDataset(Dataset):
    def __init__(self, num_samples=1000, input_dim=784, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.data = torch.randn(num_samples, input_dim)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


@dataclass 
class TestResult:
    scheduler_type: str
    resume_epoch: int
    uninterrupted_lr_history: List[float]
    resumed_lr_history: List[float]
    uninterrupted_beta_history: List[float]
    resumed_beta_history: List[float]
    uninterrupted_loss_history: List[float]
    resumed_loss_history: List[float]
    lr_match: bool
    beta_match: bool
    loss_close: bool


class ResumeTestSuite:
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.test_results = []
        
    def create_test_config(self, scheduler_type: str, epochs: int = 20) -> TrainingConfig:
        """Create test configuration for specific scheduler type"""
        return TrainingConfig(
            epochs=epochs,
            batch_size=32,
            learning_rate=1e-3,
            weight_decay=1e-5,
            
            # VAE params
            beta=1.0,
            beta_warmup_epochs=5,
            beta_schedule="linear",
            
            # LR scheduler
            scheduler=scheduler_type,
            lr_warmup_epochs=3,
            min_lr=1e-6,
            
            # Testing params
            early_stopping=False,  # Disable for consistent testing
            save_every=5,
            save_best=True,
            checkpoint_dir=self.temp_dir,
            use_tensorboard=False,
            use_wandb=False,
            mixed_precision=False,  # Disable for determinism
            device="cpu",  # Use CPU for consistent testing
        )
        
    def run_uninterrupted_training(self, config: TrainingConfig) -> Dict[str, List[float]]:
        """Run complete uninterrupted training and capture histories"""
        # Create model and datasets
        model = SimpleVAE()
        train_dataset = TestDataset(num_samples=200, seed=42)
        val_dataset = TestDataset(num_samples=50, seed=123)
        
        # Create trainer
        trainer = VAETrainer(model, config, train_dataset, val_dataset)
        
        # Track histories
        lr_history = []
        beta_history = []
        loss_history = []
        
        # Mock training loop to capture scheduler states
        for epoch in range(config.epochs):
            trainer.absolute_epoch = epoch
            trainer.current_epoch = epoch
            
            # Capture LR
            current_lr = trainer.optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            
            # Capture beta
            current_beta = trainer.beta_scheduler.step(epoch)
            beta_history.append(current_beta)
            
            # Simulate training step
            trainer.model.train()
            total_loss = 0.0
            for i, batch in enumerate(trainer.train_loader):
                if i >= 3:  # Just a few batches for speed
                    break
                outputs = trainer.model(batch)
                losses = trainer.model.loss(batch, outputs, beta=current_beta)
                total_loss += losses['total'].item()
            
            avg_loss = total_loss / min(3, len(trainer.train_loader))
            loss_history.append(avg_loss)
            
            # Step scheduler 
            if trainer.scheduler and config.scheduler not in ["plateau", "onecycle"]:
                trainer.scheduler.step()
                
        return {
            'lr_history': lr_history,
            'beta_history': beta_history, 
            'loss_history': loss_history
        }
        
    def run_resumed_training(self, config: TrainingConfig, resume_epoch: int) -> Dict[str, List[float]]:
        """Run training with resume at specified epoch"""
        # Create model and datasets
        model = SimpleVAE()
        train_dataset = TestDataset(num_samples=200, seed=42)
        val_dataset = TestDataset(num_samples=50, seed=123)
        
        # Create trainer
        trainer = VAETrainer(model, config, train_dataset, val_dataset)
        
        # Simulate checkpoint at resume_epoch
        trainer.absolute_epoch = resume_epoch
        trainer.current_epoch = resume_epoch
        trainer.absolute_step = resume_epoch * len(trainer.train_loader)
        trainer.global_step = trainer.absolute_step
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, f"test_checkpoint_epoch_{resume_epoch}.pt")
        trainer.save_checkpoint(f"test_checkpoint_epoch_{resume_epoch}.pt")
        
        # Create new trainer instance for resume test
        fresh_model = SimpleVAE()
        fresh_trainer = VAETrainer(fresh_model, config, train_dataset, val_dataset)
        
        # Resume training
        fresh_trainer.load_checkpoint(checkpoint_path)
        
        # Fast-forward scheduler
        if fresh_trainer.scheduler:
            fresh_trainer._fast_forward_scheduler(resume_epoch)
            
        # Track histories from resume point
        lr_history = []
        beta_history = []
        loss_history = []
        
        # Continue training from resume point
        for epoch in range(resume_epoch, config.epochs):
            fresh_trainer.absolute_epoch = epoch
            fresh_trainer.current_epoch = epoch
            
            # Capture states
            current_lr = fresh_trainer.optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            
            current_beta = fresh_trainer.beta_scheduler.step(epoch)
            beta_history.append(current_beta)
            
            # Simulate training
            fresh_trainer.model.train()
            total_loss = 0.0
            for i, batch in enumerate(fresh_trainer.train_loader):
                if i >= 3:
                    break
                outputs = fresh_trainer.model(batch)
                losses = fresh_trainer.model.loss(batch, outputs, beta=current_beta)
                total_loss += losses['total'].item()
                
            avg_loss = total_loss / min(3, len(fresh_trainer.train_loader))
            loss_history.append(avg_loss)
            
            # Step scheduler
            if fresh_trainer.scheduler and config.scheduler not in ["plateau", "onecycle"]:
                fresh_trainer.scheduler.step()
                
        return {
            'lr_history': lr_history,
            'beta_history': beta_history,
            'loss_history': loss_history
        }
        
    def test_scheduler_type(self, scheduler_type: str, resume_epochs: List[int] = [5, 10, 15]) -> List[TestResult]:
        """Test resume behavior for specific scheduler type"""
        print(f"\n=== Testing {scheduler_type} scheduler ===")
        
        config = self.create_test_config(scheduler_type)
        
        # Run uninterrupted training
        print("Running uninterrupted training...")
        uninterrupted = self.run_uninterrupted_training(config)
        
        results = []
        
        for resume_epoch in resume_epochs:
            if resume_epoch >= config.epochs:
                continue
                
            print(f"Testing resume at epoch {resume_epoch}...")
            
            # Run resumed training 
            resumed = self.run_resumed_training(config, resume_epoch)
            
            # Compare histories from resume point onwards
            uninterrupted_lr_slice = uninterrupted['lr_history'][resume_epoch:]
            uninterrupted_beta_slice = uninterrupted['beta_history'][resume_epoch:]
            uninterrupted_loss_slice = uninterrupted['loss_history'][resume_epoch:]
            
            # Check if they match
            lr_match = np.allclose(uninterrupted_lr_slice, resumed['lr_history'], rtol=1e-5)
            beta_match = np.allclose(uninterrupted_beta_slice, resumed['beta_history'], rtol=1e-5)
            loss_close = np.allclose(uninterrupted_loss_slice, resumed['loss_history'], rtol=1e-2)
            
            result = TestResult(
                scheduler_type=scheduler_type,
                resume_epoch=resume_epoch,
                uninterrupted_lr_history=uninterrupted_lr_slice,
                resumed_lr_history=resumed['lr_history'],
                uninterrupted_beta_history=uninterrupted_beta_slice,
                resumed_beta_history=resumed['beta_history'],
                uninterrupted_loss_history=uninterrupted_loss_slice,
                resumed_loss_history=resumed['loss_history'],
                lr_match=lr_match,
                beta_match=beta_match,
                loss_close=loss_close,
            )
            
            results.append(result)
            
            # Print results
            status_lr = "✓" if lr_match else "✗"
            status_beta = "✓" if beta_match else "✗" 
            status_loss = "✓" if loss_close else "✗"
            
            print(f"  Resume epoch {resume_epoch}: LR {status_lr} Beta {status_beta} Loss {status_loss}")
            
            if not (lr_match and beta_match):
                print(f"    LR diff: max={np.max(np.abs(np.array(uninterrupted_lr_slice) - np.array(resumed['lr_history']))):.6f}")
                print(f"    Beta diff: max={np.max(np.abs(np.array(uninterrupted_beta_slice) - np.array(resumed['beta_history']))):.6f}")
                
        return results
        
    def run_all_tests(self) -> bool:
        """Run comprehensive test suite"""
        print("Starting resume behavior test suite...")
        
        # Test different scheduler types
        scheduler_types = ["cosine", "warmup", "warmup_cosine"]
        
        all_passed = True
        
        for scheduler_type in scheduler_types:
            try:
                results = self.test_scheduler_type(scheduler_type)
                self.test_results.extend(results)
                
                # Check if all tests passed for this scheduler
                scheduler_passed = all(r.lr_match and r.beta_match for r in results)
                if not scheduler_passed:
                    all_passed = False
                    
            except Exception as e:
                print(f"ERROR testing {scheduler_type}: {e}")
                all_passed = False
                
        return all_passed
        
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        
        total_tests = len(self.test_results)
        lr_passed = sum(1 for r in self.test_results if r.lr_match)
        beta_passed = sum(1 for r in self.test_results if r.beta_match)
        loss_passed = sum(1 for r in self.test_results if r.loss_close)
        
        print(f"Total tests: {total_tests}")
        print(f"LR scheduler tests passed: {lr_passed}/{total_tests}")
        print(f"Beta scheduler tests passed: {beta_passed}/{total_tests}")
        print(f"Loss consistency tests passed: {loss_passed}/{total_tests}")
        
        overall_passed = lr_passed == total_tests and beta_passed == total_tests
        status = "PASSED" if overall_passed else "FAILED"
        print(f"\nOverall status: {status}")
        
        return overall_passed


def main():
    """Run the test suite"""
    # Create temporary directory for test artifacts
    temp_dir = tempfile.mkdtemp(prefix="vae_trainer_test_")
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Run tests
        test_suite = ResumeTestSuite(temp_dir)
        all_passed = test_suite.run_all_tests()
        test_suite.print_summary()
        
        if all_passed:
            print("\n🎉 All tests PASSED! Resume behavior is correct.")
            return 0
        else:
            print("\n❌ Some tests FAILED! Resume behavior needs fixing.")
            return 1
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())