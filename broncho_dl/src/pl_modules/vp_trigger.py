import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import logging

logger = logging.getLogger(__name__)

class TriggerPredictor(pl.LightningModule):
    def __init__(self, mdl: nn.Module, optimizer, scheduler,
                 train_loader, val_loader, test_loader, normalizer,
                 **kwargs):
        """
        Initialize the TriggerPredictor with model components and data loaders.
        
        Args:
            mdl (nn.Module): The neural network model.
            optimizer: The optimizer for training.
            scheduler: The learning rate scheduler.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            test_loader: DataLoader for test data.
            normalizer: Object to normalize/denormalize predictions and ground truth.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.mdl = mdl
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.normalizer = normalizer
        # Lists to store outputs for metric computation across batches
        self.validation_epoch_outputs = []
        self.test_epoch_outputs = []
        # We use manual optimization.
        self.automatic_optimization = False

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def _calculate_loss(self, batch, mode="train"):
        """
        Calculate the loss for a given batch.
        
        The training now predicts the trigger (or score) rather than a velocity vector.
        The ground truth trigger is taken from the batch key "trigger" (shape [B]).
        In addition to the MSE loss, during validation we also compute a binary classification
        accuracy where predictions (and gt) are thresholded at 0.8.
        
        Args:
            batch: The input batch containing:
                - "rgb_raw": Tensor of shape [B, seq_len, 3, 224, 224].
                - "current_area_seq": Tensor of shape [B, seq_len] (trajectory indices).
                - "target_area_seq": Tensor of shape [B, seq_len] (trajectory indices).
                - "bend_vel", "rot_vel", "trans_vel": each Tensor of shape [B, seq_len],
                   these are used to compute the velocity input.
                - "trigger": A scalar value per sample (ground truth trigger/score).
            mode (str): Mode of operation ("train", "val", or "test").
        
        Returns:
            dict: Dictionary with computed loss and metrics.
        """
        total_loss = 0
        metrics = {}
        
        # --- Build model inputs ---
        # Normalize the image sequence.
        # Expected shape: [B, seq_len, 3, 224, 224]
        rgb = batch["rgb_raw"]
        rgb = self.normalizer.normalize(rgb, "image")
        
        # Use the area sequences directly (used as indices for embeddings).
        cur_area = batch["current_area_seq"]
        target_area = batch["target_area_seq"]
        
        # Build velocity input by stacking bend, rot, and trans velocities.
        # Each velocity component has shape [B, seq_len]; unsqueeze to [B, seq_len, 1].
        bend_vel = self.normalizer.normalize(batch["bend_vel"].unsqueeze(-1), "Bend_vel")
        rot_vel  = self.normalizer.normalize(batch["rot_vel"].unsqueeze(-1), "Rot_vel")
        trans_vel = self.normalizer.normalize(batch["trans_vel"].unsqueeze(-1), "Trans_vel")
        # Stack to obtain a tensor of shape [B, seq_len, 3]
        vel = torch.cat([bend_vel, rot_vel, trans_vel], dim=-1)
        
        # The ground truth trigger is provided by the batch, shape [B].
        # Expand it to shape [B, 1] to match the network output.
        gt = batch["trigger"].unsqueeze(-1)
        
        # Pass the inputs to the model. Note: even though the model still expects
        # a velocity input, the prediction task now is to estimate the trigger.
        out = self.mdl(rgb, cur_area, target_area, vel)  # Expected shape: [B, 1]
        
        # --- Compute Mean Squared Error loss on the trigger prediction ---
        l2 = F.mse_loss(out, gt, reduction='none').mean()
        metrics['l2_loss'] = l2
        total_loss += l2

        if mode == "train":
            # Log learning rate from scheduler.
            self.log("train_learning_rate", self.scheduler.get_last_lr()[0],
                     on_step=True, prog_bar=True)
        else:
            # For evaluation we compute additional regression metrics and a binary accuracy.
            diff = out - gt
            abs_error = torch.abs(diff)
            batch_mae = abs_error.mean().item()  # Mean Absolute Error
            
            metrics['batch_mae'] = batch_mae

            # --- Compute binary classification accuracy ---
            # Apply threshold 0.8: values >= 0.8 are quantized to 1, others to 0.
            threshold = 0.8
            pred_class = (out >= threshold).float()
            gt_class = (gt >= threshold).float()
            binary_acc = (pred_class == gt_class).float().mean().item()
            metrics['binary_accuracy'] = binary_acc

        metrics["total_loss"] = total_loss
        mod_metric = {f"{mode}_{key}": value for key, value in metrics.items()}
        self.log_dict(mod_metric)
        return metrics

    def train_dataloader(self):
        """Return the training DataLoader."""
        return self.train_loader

    def val_dataloader(self):
        """Return the validation DataLoader."""
        return self.val_loader

    def test_dataloader(self):
        """Return the test DataLoader."""
        return self.test_loader

    def training_step(self, batch, batch_idx):
        """Perform a training step with manual optimization."""
        opt = self.optimizers()
        sch = self.lr_schedulers()
        train_step_output = self._calculate_loss(batch, mode="train")
        opt.zero_grad()
        self.manual_backward(train_step_output["total_loss"])
        opt.step()
        sch.step()

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step, computing loss and evaluation metrics.
        
        Args:
            batch: The input batch.
            batch_idx (int): Index of the batch.
        """
        val_step_output = self._calculate_loss(batch, mode="val")
        self.validation_epoch_outputs.append(val_step_output)

    def on_validation_epoch_end(self):
        """Aggregate validation metrics at the end of an epoch."""
        outputs = self.validation_epoch_outputs

        # For trigger prediction, we compute regression metrics.
        all_mae = [x['batch_mae'] for x in outputs]
        overall_mae = sum(all_mae) / len(all_mae) if all_mae else 0

        # Also average the binary accuracy.
        all_binary_acc = [x['binary_accuracy'] for x in outputs]
        overall_binary_acc = sum(all_binary_acc) / len(all_binary_acc) if all_binary_acc else 0

        self.log("val_mae", overall_mae, on_epoch=True)
        self.log("val_accu", overall_binary_acc, on_epoch=True)

        # Also average the validation loss.
        all_val_loss = [x['l2_loss'] for x in outputs]
        avg_val_loss = sum(all_val_loss) / len(all_val_loss)
        self.log("val_loss", avg_val_loss, on_epoch=True)

        # Clear outputs for next epoch.
        self.validation_epoch_outputs.clear()
