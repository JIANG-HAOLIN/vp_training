import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import sys
import torch.nn as nn
import torch
from broncho_dl.src.trainer_util import launch_trainer
from datetime import datetime
import pathlib

log = logging.getLogger(__name__)


def set_random_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


@hydra.main(config_path=str(pathlib.Path(__file__).parent.joinpath(
        'broncho_dl','configs')), config_name='', version_base='1.2')
def train(cfg: DictConfig) -> None:
    # set_random_seed(42)
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    torch.set_float32_matmul_precision('medium')
    project_path = os.path.abspath(os.path.join(__file__, '..'))
    hydra_cfg_og = HydraConfig.get()
    multirun_dir_path = hydra_cfg_og.sweep.dir

    log.info('*-------- train func starts --------*')
    log.info('output folder:' + multirun_dir_path + '\n')
    log.info('project_path:' + project_path + '\n')
    sys.path.append(project_path)
    log.info('sys.path:', )
    for p in sys.path:
        log.info(p)

    log.info(f"if Cuda available:{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"Cuda info:\n{torch.cuda.get_device_properties('cuda')}")
        log.info(f"Cuda version:{torch.version.cuda}")
        device = "cuda"
    else:
        log.info(f'no Cuda detected, using CPU instead !!')
        device = "cpu"
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    log.info(f"Current Project path: {project_path}")
    log.info(f"current multi-run output path: {multirun_dir_path}")

    from broncho_dl.utils.hydra_utils import extract_sweeper_output_label
    label = extract_sweeper_output_label(cfg, hydra_cfg_og.runtime.choices)
    log.info(f"current running output label: {label}")
    out_dir_path = os.path.join(multirun_dir_path, label + '_' + datetime.now().strftime("%m-%d-%H:%M:%S"))
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    log.info(f"current experiment output path: {out_dir_path}")

    model: nn.Module = hydra.utils.instantiate(cfg.models.model, _recursive_=False).to(device)
    log.info(f"model trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    log.info(f"model non-trainable params:{sum(p.numel() for p in model.parameters() if not p.requires_grad)}")
    train_loader, val_loader, test_loader, normalizer, train_dataset, val_dataset = hydra.utils.instantiate(cfg.datasets.dataloader, project_path=project_path,
                                                                    save_json=out_dir_path)

    optimizer = hydra.utils.instantiate(cfg.optimizers.optimizer, params=model.parameters())

    with open_dict(cfg):
        cfg.optimizers.scheduler.num_training_steps = len(train_loader) * cfg.trainers.launch_trainer.max_epochs
        cfg.optimizers.scheduler.num_warmup_steps = int(
            len(train_loader) * cfg.trainers.launch_trainer.max_epochs * 0.01)
    lr_scheduler = hydra.utils.instantiate(cfg.optimizers.scheduler, optimizer=optimizer)
    pl_module = hydra.utils.instantiate(cfg.pl_modules.pl_module, model,
                                        optimizer, lr_scheduler,
                                        train_loader, val_loader, test_loader, normalizer, _recursive_=False)

    with open(os.path.join(out_dir_path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    launch_trainer(pl_module, out_dir_path=out_dir_path, label=label, hydra_conf=cfg,
                   model_name=cfg.models.name, dataset_name=cfg.datasets.name, task_name=cfg.task_name,
                   **cfg.trainers.launch_trainer)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    train()
