#!/usr/bin/env python3
import logging

import lightning as L
import torch
import torch.nn.functional as F

from pipeline.optim.optimizer_registry import create_optimizer
from pipeline.optim.scheduler_registry import create_scheduler

logger = logging.getLogger(__name__)


class MEGUnsupervisedPretrainer(L.LightningModule):
    """Lightning module for unsupervised pretraining of BIOT models using contrastive learning.

    This module implements contrastive learning to pretrain the BIOT encoder,
    which can later be used in a supervised model.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            config: dict,
            temperature: float = 0.1,
    ):
        """Initialize the unsupervised pretrainer.

        Args:
            model: The UnsupervisedPretrain model
            config: Training configuration
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()
        self.model = model
        self.config = config
        self.temperature = temperature
        self.save_hyperparameters(config)
        
        # Store optimizer and scheduler configs for later use
        self.optimizer_config = config["optimizer"]
        self.scheduler_config = config.get("scheduler", None)
        logger.info(f"Optimizer config: {self.optimizer_config}")
        if self.scheduler_config:
            logger.info(f"Scheduler config: {self.scheduler_config}")
        
    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, n_windows, n_channels, n_samples_per_window)

        Returns:
            Tuple of embeddings and predicted embeddings
        """
        # The UnsupervisedPretrain model returns (embedding, predicted_embedding)
        return self.model(x)

    def contrastive_loss(self, emb, pred_emb):
        """Compute contrastive loss between embeddings.

        Args:
            emb: Embeddings from perturbed input
            pred_emb: Embeddings from unperturbed input

        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        emb = F.normalize(emb, dim=1)
        pred_emb = F.normalize(pred_emb, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(emb, pred_emb.T) / self.temperature

        # Labels are the diagonal indices (positive pairs)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        # Compute cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            Training loss
        """
        # Extract data and ignore labels for unsupervised learning
        x, _ = batch
        
        # The unsupervised take single window input
        batch_size, n_windows, n_channels, n_samples_per_window = x.shape
        losses = []
        for i in range(n_windows):
            # Extract single window input
            x_window = x[:, i, :, :]
            # Forward pass
            emb, pred_emb = self(x_window)

            # Compute contrastive loss
            loss_window = self.contrastive_loss(emb, pred_emb)
            losses.append(loss_window)
            
            if loss_window.isnan().any():
                logger.warning(
                    f"NaN loss detected in window {i} of batch {batch_idx}. "
                    "This may indicate issues with the input data or model."
                )
            if loss_window.isinf().any():
                logger.warning(
                    f"Inf loss detected in window {i} of batch {batch_idx}. "
                    "This may indicate issues with the input data or model."
                )
            if loss_window <= 0:
                logger.warning(
                    f"Negative or null loss detected in window {i} of batch {batch_idx}. "
                    "This may indicate issues with the input data or model."
                )
            
        # Average loss across windows
        loss = torch.mean(torch.stack(losses))
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        # Extract data and ignore labels for unsupervised learning
        x, _ = batch

        # The unsupervised take single window input
        batch_size, n_windows, n_channels, n_samples_per_window = x.shape
        losses = []
        for i in range(n_windows):
            # Extract single window input
            x_window = x[:, i, :, :]
            # Forward pass
            emb, pred_emb = self(x_window)
            
            # Compute contrastive loss
            loss_window = self.contrastive_loss(emb, pred_emb)
            losses.append(loss_window)
        # Average loss across windows
        loss = torch.mean(torch.stack(losses))        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        logger.info("Configuring optimizers and schedulers")
        
        # Create optimizer
        optimizer = create_optimizer(self.optimizer_config, self.model.parameters())
        
        if self.scheduler_config is None:
            logger.info("No scheduler configured, using optimizer only")
            return optimizer
        
        # Create scheduler using registry
        scheduler = create_scheduler(self.scheduler_config, optimizer)
        
        # Handle ReduceLROnPlateau special case
        scheduler_name = self.scheduler_config.get("name")
        if scheduler_name == "ReduceLROnPlateau":
            logger.info("Using ReduceLROnPlateau with val_loss monitoring")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_config.get("monitor", "val_loss"),
                    "frequency": self.scheduler_config.get("frequency", 1),  # Default to every epoch
                    "interval": "epoch"   # ReduceLROnPlateau works per epoch
                }
            }
        
        logger.info("Optimizer and scheduler configured successfully")
        return [optimizer], [scheduler]
