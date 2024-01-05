from ast import Not
import logging
import hydra
import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from omegaconf import OmegaConf
from ..models.metric import NERMetric, RelationMetric
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup,\
                                get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
log = logging.getLogger(__name__)

class Runner(pl.LightningModule):
    def __init__(self, cfg, fields, **kwargs):
        super(Runner, self).__init__()
        
        self.cfg = cfg 
        self.fields = fields
        
        self.model_cfg = cfg.model
        self.optim_cfg = cfg.optim
        
        self.metric_dev_hist = [None]
        self.metric_test_hist = [None]
        
        self.best_dev_metric_epoch = -1
        self.save_hyperparameters(OmegaConf.to_container(self.cfg, resolve=True))
        self.test = False
                
    def forward(self, x, y, inference = False):
        return self.model(x, y, inference)
        
    # @property
    # def result(self):
    #     raise NotImplementedError
    
    # def on_fit_start(self):
    #     raise NotImplementedError
        
    # def on_train_epoch_end(self, outputs):
    #     raise NotImplementedError
    
    # def training_step(self, batch, batch_idx):
    #     raise NotImplementedError
    
    # def on_validation_epoch_start(self):
    #     raise NotImplementedError
        
    # def on_validation_epoch_end(self, validation_step_outputs):
    #     raise NotImplementedError
    
    # def validation_step(self, batch, batch_idx):
    #     raise NotImplementedError
        
    # def test_step(self, batch, batch_idx):
    #     raise NotImplementedError
        
    # def on_test_epoch_start(self):
    #     raise NotImplementedError
        
    # def on_test_epoch_end(self, test_step_outputs):
    #     raise NotImplementedError
        
    @property     
    def num_training_steps(self) -> int:
        """
        https://github.com/PyTorchLightning/pytorch-lightning/issues/5449#issuecomment-757863689
        Total training steps inferred from datamodule and devices.
        """
        dataset = self.trainer.datamodule.train_dataloader()
        
        if self.trainer.max_steps and self.trainer.max_steps != -1:
            return self.trainer.max_steps
    
        if self.trainer.limit_train_batches != 0:
            if isinstance(self.trainer.limit_train_batches, float):
                dataset_size = self.trainer.limit_train_batches * len(dataset)
            elif isinstance(self.trainer.limit_train_batches, int):
                dataset_size = self.trainer.limit_train_batches
        else:
            dataset_size = len(dataset)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs
        
    def configure_optimizers(self):
        log.info(f"total tarining steps: {self.num_training_steps}")
        hparams = self.optim_cfg
        # pdb.set_trace()
        # for n, c in self.model.named_children():
        #     print(n)
        #     print(c)
        # lr_rate: 用来放大encoder的learning rate， 如果存在，那么用的就是finetuning的模式。
        if hparams.get("lr_rate") is not None:

            if hparams.only_embeder:
                optimizer = AdamW(
                    [{'params': c.parameters(), 'lr': hparams.lr * (1 if n == 'embedding' else hparams.lr_rate)}
                     for n, c in self.model.named_children()], hparams.lr
                )

                log.info(f"Embeddings has learning rate:{  hparams.lr},Encoder has learning rate:{hparams.lr * hparams.lr_rate}, Scorer has lr:{hparams.lr * hparams.lr_rate}"
                         )

            else:
                optimizer = torch.optim.Adam(
                    [{'params': c.parameters(), 'lr': hparams.lr * (1 if n == 'embedding' or 'encoder' in n else hparams.lr_rate)}
                     for n, c in self.model.named_children()], hparams.lr, betas=(hparams.beta1, hparams.beta2)
                )
                log.info(f"Embeddings has learning rate:{hparams.lr},Encoder has learning rate:{hparams.lr}, Scorer has lr:{hparams.lr * hparams.lr_rate}"
                         )

            if hparams.scheduler_type == 'linear_warmup':
                # scheduler = get_linear_schedule_with_warmup(optimizer, 211231123, 31212312312)
                scheduler = get_linear_schedule_with_warmup(optimizer, 
                                hparams.warmup * self.num_training_steps, self.num_training_steps)
                scheduler = {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
                log.info("Using huggingface transformer linear-warmup scheduler.")

            elif hparams.scheduler_type == 'constant_warmup':
                scheduler = get_constant_schedule_with_warmup(optimizer, hparams.warmup * self.num_training_steps)
                scheduler = {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
                log.info("Using huggingface transformer constant_warmup scheduler.")
                
            elif hparams.scheduler_type == 'cosine_warmup':
                scheduler = get_cosine_schedule_with_warmup(optimizer, hparams.warmup * self.num_training_steps, self.num_training_steps, hparams.num_cycles)
                scheduler = {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
                log.info("Using huggingface transformer cosine_warmup scheduler.")
                
            elif hparams.scheduler_type == 'cosine_hard_warmup':
                scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, hparams.warmup * self.num_training_steps,
                                                                                self.num_training_steps, hparams.num_cycles)
                scheduler = {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
                log.info("Using huggingface transformer cosine_with_hard_restarts_warmup scheduler.")
                
            elif hparams.scheduler_type == 'reduce_on_plateau':
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = hparams.factor, patience = hparams.patience)
                scheduler = {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'monitor': 'val_checkpoint_on'
                }
                log.info("Using pytorch reduce_on_plateau scheduler.")
            
            return [optimizer], [scheduler]
        else:
            opt = hydra.utils.instantiate(
                hparams.optimizer, params=self.parameters(), _convert_='all'
            )

            if hparams.use_lr_scheduler:
                if hparams.lr_scheduler._target_ == 'torch.optim.lr_scheduler.ExponentialLR':
                    scheduler =  torch.optim.lr_scheduler.ExponentialLR(opt, gamma=.75 ** (1 / 5000))
                    scheduler = {
                        'scheduler': scheduler,
                        'interval': 'step',  # or 'epoch'
                        'frequency': 1
                    }
                    log.info("Using ExponentialLR")

                else:
                    raise NotImplementedError

                return [opt], [scheduler]
            return opt
    
    



