from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks import Callback
from lightning import Trainer
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


class MyEpochTimer(Callback):
    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        print(self.epoch_start_time)

    def on_epoch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.epoch_start_time
        print(f"Epoch {trainer.current_epoch + 1} took {elapsed_time:.2f} seconds")


class SaveBestTxt(Callback):
    """this hook is always executed before the main code defined in pl_modules, so at epoch n only the best_model_score
    only contains the best value appeared in epoch 0 --> n-1 """

    def __init__(self, out_dir_path: str, label: str):
        super().__init__()
        self.out_dir_path = out_dir_path
        self.label = label

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        # Log the current best model score and path to a text file
        if not trainer.sanity_checking:
            if trainer.current_epoch > 0:
                txt_file_path = os.path.join(self.out_dir_path, f'best_{self.label}.text')
                with open(txt_file_path, 'a') as file:
                    file.write(f"Epoch: {trainer.current_epoch - 1}, "
                               f"Best Model Score: {trainer.checkpoint_callback.best_model_score:.8f}, "
                               f"Best Model Path: {trainer.checkpoint_callback.best_model_path}\n")

    def on_fit_end(self, trainer: Trainer, pl_module):
        # Log the current best model score and path to a text file
        txt_file_path = os.path.join(self.out_dir_path, f'best_{self.label}.text')
        with open(txt_file_path, 'a') as file:
            file.write(f"Epoch: {trainer.current_epoch - 1}, "
                       f"Best Model Score: {trainer.checkpoint_callback.best_model_score:.8f}, "
                       f"Best Model Path: {trainer.checkpoint_callback.best_model_path}\n")


class PlotMetric(Callback):
    def __init__(self, out_dir_path: str,
                 wanted_metrics: tuple = ("loss", "acc", "max_ae", "rmse", "nrmse"),
                 freq: int = 20,
                 ):
        super().__init__()
        self.out_dir_path = out_dir_path
        self.wanted_metrics = wanted_metrics
        self.val_metrics = {}
        self.train_metrics = {}
        self.freq = freq

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        epoch = trainer.current_epoch
        if trainer.sanity_checking:
            return

        # ——— ALWAYS store metrics every epoch ———
        for metric, value in trainer.callback_metrics.items():
            v = value.detach().cpu().numpy()
            if "val" in metric and any(w in metric for w in self.wanted_metrics):
                self.val_metrics.setdefault(metric, []).append(v)
            if "train" in metric and any(w in metric for w in self.wanted_metrics):
                self.train_metrics.setdefault(metric, []).append(v)

        # ——— only print/plot every `freq` epochs ———
        if epoch % self.freq != 0:
            return

        # print history
        # print(f"\n=== Epoch {epoch} ===")
        # print("Validation metrics history:")
        # for m, vals in self.val_metrics.items():
        #     print(f"  {m}: {vals}")
        # print("Training metrics history:")
        # for m, vals in self.train_metrics.items():
        #     print(f"  {m}: {vals}")

        # plotting (unchanged except moved here)
        # — validation plot —
        num_val = len(self.val_metrics)
        if num_val:
            fig, axes = plt.subplots(num_val, 1, figsize=(5, 3 * num_val))
            for idx, (metric, values) in enumerate(self.val_metrics.items()):
                x = np.arange(len(values)); y = np.asarray(values)
                ax = axes[idx] if num_val > 1 else axes
                ax.plot(x, y, '-', label=metric, linewidth=0.2)
                ax.set_xlabel('epoch_chunk'); ax.set_ylabel(metric)
                best_pos = (np.argmax(y) if "acc" in metric else np.argmin(y))
                ax.set_title(f'{metric} = {y[best_pos]:.4f} at chunk {best_pos}')
            fig.savefig(os.path.join(self.out_dir_path, 'val_metrics.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

        # — training plot —
        num_train = len(self.train_metrics)
        if num_train:
            fig, axes = plt.subplots(num_train, 1, figsize=(5, 3 * num_train))
            for idx, (metric, values) in enumerate(self.train_metrics.items()):
                x = np.arange(len(values)); y = np.asarray(values)
                ax = axes[idx] if num_train > 1 else axes
                ax.plot(x, y, '-', label=metric, linewidth=0.2)
                ax.set_xlabel('epoch_chunk'); ax.set_ylabel(metric)
                best_pos = (np.argmax(y) if "acc" in metric else np.argmin(y))
                ax.set_title(f'{metric} = {y[best_pos]:.4f} at chunk {best_pos}')
            fig.savefig(os.path.join(self.out_dir_path, 'train_metrics.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)



class NaNCallback(Callback):

    def nan_detect(self, trainer, pl_moudle, x, name):
        if isinstance(x, torch.Tensor):
            if torch.isnan(x).any():
                print(f"Nan detected in {name}")
                trainer.should_stop = True
        elif isinstance(x, dict):
            for k, v in x.items():
                self.nan_detect(trainer, pl_moudle, v, name + k)
        elif isinstance(x, list):
            for v in x:
                self.nan_detect(trainer, pl_moudle, v, name)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.nan_detect(trainer, pl_module, outputs, "output")
        self.nan_detect(trainer, pl_module, batch, "batch input")
