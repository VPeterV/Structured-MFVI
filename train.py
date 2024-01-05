import os
import pdb
import shutil
from pathlib import Path
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
import wandb
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

# hydra imports
from omegaconf import DictConfig
from hydra.utils import log
import hydra
from omegaconf import OmegaConf

# normal imports
from typing import List
import warnings
import logging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import sys
# sys.setdefaultencoding() does not exist, here!
from omegaconf import OmegaConf, open_dict
from src.utils.helpers import assign_free_gpus


def train(config):
    
    log.info(OmegaConf.to_container(config,resolve=True))
    config.root = hydra.utils.get_original_cwd()

    hydra_dir = str(os.getcwd())
    seed_everything(config.seed)
    os.chdir(hydra.utils.get_original_cwd())

    # Instantiate datamodule
    hydra.utils.log.info(os.getcwd())
    hydra.utils.log.info(f"Instantiating <{config.datamodule.target}>")
    # Instantiate callbacks and logger.
    callbacks: List[Callback] = []
    logger: List[LightningLoggerBase] = []

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config.datamodule.target, config.datamodule, _recursive_=False
    )
    
    # check
    if config.datamodule.use_bert and 'large' in config.datamodule.bert and config.model.embeddings.n_bert_out != 1024:
        # raise ValueError
        log.info("Attention !!! The dimension of embedding is not 1024 !!!")
    
    datamodule.build_data()
    log.info("created datamodule")
    model = hydra.utils.instantiate(config.runner, cfg = config, fields=datamodule.fields, _recursive_=False)
    
    os.chdir(hydra_dir)
    
# Train the model ⚡
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    if config.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                monitor=f'{config.task}/valid/score',
                mode='max',
                save_last=False,
                filename='checkpoint',
                dirpath=config.checkpoint_path
            )
        )
        log.info("Instantiating callback, ModelCheckpoint")
        
    if config.patience > 0:
        callbacks.append(EarlyStopping(f'{config.task}/valid/score', mode='max',
                            patience=config.patience))
                            
    if config.optim.scheduler_type == 'reduce_on_plateau':
        callbacks.append(LearningRateMonitor(logging_interval = 'epoch'))

    if config.use_logger and not config.wandb:
        logger.append(hydra.utils.instantiate(config.logger.csv))

    if config.wandb:
        if not os.path.exists('./wandb'):
            os.mkdir('./wandb', mode=0o777)
            
        config.logger.wandb.name = config.model.name
        logger.append(hydra.utils.instantiate(config.logger.wandb))
        
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    
    if config.mixed_precision:
        log.info("Mixed-Precision Training Turned On")
        precision = 16
        accelerator = 'gpu'
    else:
        precision = 32
        accelerator = None
            
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger,
        replace_sampler_ddp=False,
        precision = precision,
        accelerator = accelerator,
        gpus = [config.device],
        accumulate_grad_batches=config.accumulation,
        checkpoint_callback=config.checkpoint,
        num_sanity_val_steps=0,
    )

    log.info(f"Starting training!")
    if config.wandb:
        logger[-1].experiment.save(str(hydra_dir) + "/.hydra/*", base_path=str(hydra_dir))

    trainer.fit(model, datamodule)
    log.info(f"Finalizing!")
    #TODO fast dev
    # trainer.test(model, datamodule = datamodule)
    trainer.test(model, datamodule = datamodule, ckpt_path="best")
    log.info(f"Test Finished!")

    log.info(f'hydra_path:{os.getcwd()}')
    
    if "conll05" in datamodule.conf.name.lower():
        from fastNLP.core import SequentialSampler, DataSetIter
            
        sampler = SequentialSampler()
            
        ood_dataloader = DataSetIter(datamodule.datasets['test_ood'], batch_size = datamodule.conf.batch_size, 
                                sampler=sampler, num_workers=4)
        
        log.info("OOD Test Begin")
        trainer.test(model, dataloaders=ood_dataloader)
        log.info("OOD Test Finished")
    
    if config.wandb:
        logger[-1].experiment.save(str(hydra_dir) + "/*.log", base_path=str(hydra_dir))
        wandb.finish()

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):
    config.device = assign_free_gpus()
    train(config)    # return for optuna

if __name__ == "__main__":
    main()


