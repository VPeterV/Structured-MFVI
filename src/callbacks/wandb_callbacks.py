import subprocess
from pathlib import Path
from typing import List

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

def get_wandb_logger(trainer: pl.Trainer) -> WandbLogger:
    logger = None
    for lg in trainer.logger:
        if isinstance(lg, WandbLogger):
            logger = lg

    if not logger:
        raise Exception(
            "You are using wandb related callback,"
            "but WandbLogger was not found for some reason..."
        )

    return logger

class UploadHydraConfigFileToWandb(Callback):
    def on_fit_start(self, trainer, pl_module: pl.LightningModule) -> None:
        logger = get_wandb_logger(trainer=trainer)

        logger.experiment.save()

class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)
        

# class UploadCodeAsArtifact(Callback):
#     """Upload all code files to wandb as an artifact, at the beginning of the run."""

#     def __init__(self, code_dir: str, use_git: bool = True):
#         """

#         Args:
#             code_dir: the code directory
#             use_git: if using git, then upload all files that are not ignored by git.
#             if not using git, then upload all '*.py' file
#         """
#         self.code_dir = code_dir
#         self.use_git = use_git

#     @rank_zero_only
#     def on_train_start(self, trainer, pl_module):
#         logger = get_wandb_logger(trainer=trainer)
#         experiment = logger.experiment

#         code = wandb.Artifact("project-source", type="code")

#         if self.use_git:
#             # get .git folder path
#             git_dir_path = Path(
#                 subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
#             ).resolve()

#             for path in Path(self.code_dir).resolve().rglob("*"):

#                 # don't upload files ignored by git
#                 # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
#                 command = ["git", "check-ignore", "-q", str(path)]
#                 not_ignored = subprocess.run(command).returncode == 1

#                 # don't upload files from .git folder
#                 not_git = not str(path).startswith(str(git_dir_path))

#                 if path.is_file() and not_git and not_ignored:
#                     code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

#         else:
#             for path in Path(self.code_dir).resolve().rglob("*.py"):
#                 code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

#         experiment.log_artifact(code)


# class WatchModel(Callback):
#     """Make wandb watch model at the beginning of the run."""

#     def __init__(self, log: str = "gradients", log_freq: int = 100):
#         self.log = log
#         self.log_freq = log_freq

#     @rank_zero_only
#     def on_train_start(self, trainer, pl_module):
#         logger = get_wandb_logger(trainer=trainer)
#         logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=True)


# class LogConfusionMatrix(Callback):
#     """Generate confusion matrix every epoch and send it to wandb.
#     Expects validation step to return predictions and targets.
#     """

#     def __init__(self):
#         self.preds = []
#         self.targets = []
#         self.ready = True

#     def on_sanity_check_start(self, trainer, pl_module) -> None:
#         self.ready = False

#     def on_sanity_check_end(self, trainer, pl_module):
#         """Start executing this callback only after all validation sanity checks end."""
#         self.ready = True

#     def on_validation_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#     ):
#         """Gather data from single batch."""
#         if self.ready:
#             self.preds.append(outputs["preds"])
#             self.targets.append(outputs["targets"])

#     def on_validation_epoch_end(self, trainer, pl_module):
#         """Generate confusion matrix."""
#         if self.ready:
#             logger = get_wandb_logger(trainer)
#             experiment = logger.experiment

#             preds = torch.cat(self.preds).cpu().numpy()
#             targets = torch.cat(self.targets).cpu().numpy()

#             confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

#             # set figure size
#             plt.figure(figsize=(14, 8))

#             # set labels size
#             sn.set(font_scale=1.4)

#             # set font size
#             sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

#             # names should be uniqe or else charts from different experiments in wandb will overlap
#             experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

#             # according to wandb docs this should also work but it crashes
#             # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

#             # reset plot
#             plt.clf()

#             self.preds.clear()
#             self.targets.clear()


# class LogF1PrecRecHeatmap(Callback):
#     """Generate f1, precision, recall heatmap every epoch and send it to wandb.
#     Expects validation step to return predictions and targets.
#     """

#     def __init__(self, class_names: List[str] = None):
#         self.preds = []
#         self.targets = []
#         self.ready = True

#     def on_sanity_check_start(self, trainer, pl_module):
#         self.ready = False

#     def on_sanity_check_end(self, trainer, pl_module):
#         """Start executing this callback only after all validation sanity checks end."""
#         self.ready = True

#     def on_validation_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#     ):
#         """Gather data from single batch."""
#         if self.ready:
#             self.preds.append(outputs["preds"])
#             self.targets.append(outputs["targets"])

#     def on_validation_epoch_end(self, trainer, pl_module):
#         """Generate f1, precision and recall heatmap."""
#         if self.ready:
#             logger = get_wandb_logger(trainer=trainer)
#             experiment = logger.experiment

#             preds = torch.cat(self.preds).cpu().numpy()
#             targets = torch.cat(self.targets).cpu().numpy()
#             f1 = f1_score(targets, preds, average=None)
#             r = recall_score(targets, preds, average=None)
#             p = precision_score(targets, preds, average=None)
#             data = [f1, p, r]

#             # set figure size
#             plt.figure(figsize=(14, 3))

#             # set labels size
#             sn.set(font_scale=1.2)

#             # set font size
#             sn.heatmap(
#                 data,
#                 annot=True,
#                 annot_kws={"size": 10},
#                 fmt=".3f",
#                 yticklabels=["F1", "Precision", "Recall"],
#             )

#             # names should be uniqe or else charts from different experiments in wandb will overlap
#             experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

#             # reset plot
#             plt.clf()

#             self.preds.clear()
#             self.targets.clear()
