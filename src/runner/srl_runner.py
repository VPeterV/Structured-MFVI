import os
import logging
import pdb
import hydra
import torch
from torch import nn
# import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup,\
                                get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
# from omegaconf import OmegaConf
# from ..models.metric import NERMetric, RelationMetric
from .base import Runner
from .utils import clean_gold_labels, check_decode_mode

log = logging.getLogger(__name__)

class SRLRunner(Runner):
    def __init__(self, cfg, fields, **kwargs):
        super(SRLRunner, self).__init__(cfg, fields)
        
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.decoder_cfg = self.model_cfg.decoder
        
        label2id, self.id2label  = fields.get_span_label_dict()
        
        self.metric_cfg = self.model_cfg.metric
        self.srl_metric = hydra.utils.instantiate(self.metric_cfg.target, conf = self.metric_cfg, 
                                                        fields = fields,  _recursive_=False)
        # pdb.set_trace()
        self.model = hydra.utils.instantiate(cfg.model.target, conf = self.model_cfg, fields = fields, _recursive_ = False)
        self.decoder = hydra.utils.instantiate(self.decoder_cfg.target, conf = self.decoder_cfg, id2label = self.id2label,  _recursive_ = False)
        
        self.srl_best_dev_metric_epoch = -1
        self.srl_metric_dev_hist = []
        
        
    @property
    def srl_result(self):
        return self.srl_metric_dev_hist[self.srl_best_dev_metric_epoch]['score']

    def on_fit_start(self):
        self.srl_metric.to(self.device)
        
    def training_step(self, batch):
        # if self.current_epoch == 20:
        #     pdb.set_trace()
        x_table, y_table = batch
        result = self.model(x_table, y_table, inference = False)
        
        return result['loss']
    
    def on_validation_epoch_start(self):
        self.srl_metric.reset()
        
    def on_validation_epoch_end(self):
        mode = 'test' if self.test else 'valid'
        srl_result = self.srl_metric.compute(test=self.test, epoch_num=self.current_epoch)
        
        self.srl_metric.reset()
        
        self.log_dict({'srl' + f'/{mode}/' + k: v for k, v in srl_result.items()})
        log.info('\n')
        log.info(f'[Epoch {self.current_epoch}]\t SRL \t{mode} \t' + '\t'.join(f'{k}={v:.4f}' for k, v in srl_result.items()))

        self.update_metric(srl_results = srl_result)

            
    def validation_step(self, batch, batch_idx):
        x, y = batch
        result = self.model(x, y, inference = True)
        
        gold_srl = torch.cat([y['predicates'].unsqueeze(-1), y['gold_spans'], y['span_labels'].unsqueeze(-1)], dim = -1)
        gold_srl = gold_srl.tolist()
        gold_srl = clean_gold_labels(gold_srl, self.id2label)
        

        p_spans = None
        check_decode_mode(self.model_cfg.decoder.mode)
        
        if self.cfg.model_type == 'arc':
            if 'dp' in self.model_cfg.decoder.mode and 'argmax' in self.model_cfg.decoder.mode:
                raise NotImplementedError
                # srl_results, p_spans = self._decode_dp_argmax(x, result)
            elif self.model_cfg.decoder.mode == 'dp':
                srl_results, p_spans = self._decode_dp_arc(x, result)
        elif self.cfg.model_type == 'phpt':
            if 'dp' in self.model_cfg.decoder.mode and 'argmax' in self.model_cfg.decoder.mode:
                srl_results, p_spans = self._decode_dp_argmax(x, result)
            elif self.model_cfg.decoder.mode == 'dp':
                srl_results, p_spans = self._decode_dp(x, result)
            elif self.model_cfg.decoder.mode == 'span_repr':
                srl_results, p_spans = self._decode_treecrf_wspan(x, result)
        else:
            raise NotImplementedError
            
        # elif self.model_cfg.decoder.mode=='greedy-longest':
            # if self.current_epoch == 2:
            #     pdb.set_trace()
            # srl_results = self._decode_greedy_longest(x, result)

        
        # pdb.set_trace()
        info = {"words":x['words'],"srl":srl_results, "gold_srl": gold_srl, "p_spans":p_spans}

        self.srl_metric.update(info)
        
    def _decode_dp_arc(self, x, result):
        srl_results = []
        p_spans = []

        span_m, arc_m = result['arcs']
        labels = result['labels']

        labels_preds, arcs = self.decoder.graph2tag(arc_m, labels)
        
        # pdb.set_trace()
        for b in range(labels.shape[0]):
            srl_res, p_span = self.decoder.decode(x['words'][b], 
                            span_m[b], labels_preds[b], arcs[b], arc_m[b], self.cfg.check)
            srl_results.append(srl_res)
            p_spans.append(p_span)
            
        return srl_results, p_spans

    def _decode(self, x, result):
        
        srl_results = []

        span_m, ph_m, pt_m = result['arcs']
        span_labels = result['labels']

        spans, ph_arc, pt_arc = self.decoder.graph2tag(span_m, ph_m, pt_m, span_labels)
        
        for b in range(span_labels.shape[0]):
            srl_res = self.decoder.decode(x['words'][b], 
                            spans[b], ph_arc[b], pt_arc[b])
            srl_results.append(srl_res)
            
        return srl_results
        
    def _decode_dp(self, x, result):
        srl_results = []
        p_spans = []

        span_m, ph_m, pt_m = result['arcs']
        labels = result['labels']

        labels_preds, ph_arc, pt_arc = self.decoder.graph2tag(ph_m, pt_m, labels)
        
        # pdb.set_trace()
        for b in range(labels.shape[0]):
            srl_res, p_span = self.decoder.decode(x['words'][b], 
                            span_m[b], ph_m[b], pt_m[b], labels_preds[b], ph_arc[b], pt_arc[b], self.cfg.check)
            srl_results.append(srl_res)
            p_spans.append(p_span)
            
        return srl_results, p_spans
        
    def _decode_greedy_longest(self, x, result):
        srl_results = []

        span_m, ph_m, pt_m = result['arcs']
        span_labels = result['labels']

        span_w_labels, ph_arc, pt_arc = self.decoder.graph2tag(span_m, ph_m, pt_m, span_labels)
        
        for b in range(span_labels.shape[0]):
            srl_res = self.decoder.decode(x['words'][b], 
                            span_w_labels[b], ph_arc[b], pt_arc[b])
            srl_results.append(srl_res)
            
        return srl_results
        
    def _decode_dp_argmax(self, x, result):
        srl_results = []
        p_spans = []

        span_preds, ph_m, pt_m = result['arcs']
        ph_labels = result['labels']
        span_scores = result['scores']
        span_labels = result['span_labels']
        
        pspan = result.get('pspan', None)

        spans_batch, label_preds, ph_arc, pt_arc = self.decoder.graph2tag(span_preds, span_labels, ph_m, pt_m, ph_labels)
            
        for b in range(span_scores.shape[0]):
            if isinstance(label_preds, list):
                srl_res, p_span = self.decoder.decode(x['words'][b], 
                                span_scores[b], ph_m[b], pt_m[b], spans_batch[b], (label_preds[0][b], label_preds[1][b]), ph_arc[b], pt_arc[b])
            else:
                pspan_b = pspan if pspan is None else pspan[b]
                srl_res, p_span = self.decoder.decode(x['words'][b], 
                                span_scores[b], ph_m[b], pt_m[b], spans_batch[b], label_preds[b], ph_arc[b], pt_arc[b], pspan_b)

            srl_results.append(srl_res)
            p_spans.append(p_span)
            
        return srl_results, p_spans
        
    def _decode_treecrf_wspan(self, x, result):
        srl_results = []
        p_spans = []

        spans_batch, ph_m, pt_m = result['arcs']
        span_labels_scores = result['labels']
        span_scores = result['scores']
        span_arc_scores = result['span_arcs']

        label_preds, ph_arc, pt_arc = self.decoder.graph2tag(ph_m, pt_m, span_labels_scores)
        # breakpoint()
            
        for b in range(span_scores.shape[0]):
            srl_res, p_span = self.decoder.decode(x['words'][b], 
                            span_scores[b], span_arc_scores[b], ph_m[b], pt_m[b], spans_batch[b], label_preds[b], ph_arc[b], pt_arc[b])

            srl_results.append(srl_res)
            p_spans.append(p_span)
            
        return srl_results, p_spans
            
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
        
    def on_test_epoch_start(self):
        self.test = True
        self.on_validation_epoch_start()
        
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
        self.test = False
        
    def update_metric(self, srl_results):
        test = self.test
        if not test:
            self.srl_metric_dev_hist.append(srl_results)
            
            score = srl_results['score']
            best_score = self.srl_metric_dev_hist[self.best_dev_metric_epoch]['score']

                        
            if score >= best_score:
                self.best_dev_metric_epoch = self.current_epoch
                if self.metric_cfg.write_result_to_file:
                    os.system(f"cp {self.srl_metric.prefix}_output_valid.json  {self.srl_metric.prefix}_output_best_valid.json")
            
            log.info('\n')
            log.info(
            f'[best srl dev: {self.best_dev_metric_epoch}]\t DEV \t' + '\t'.join(f'{k}={v:.4f}'
                                            for k, v in self.srl_metric_dev_hist[self.best_dev_metric_epoch].items())
            )

        else:
            srl_best_result = srl_results
        
            log.info('\n')
            log.info(
                f'[best srl test: {self.best_dev_metric_epoch}]\t TEST \t' + '\t'.join(f'{k}={v:.4f}'
                                                for k, v in srl_best_result.items())
                )

            self.log_dict({'best_epoch': self.best_dev_metric_epoch})
            self.log_dict({f'srl/final/' + k: v for k, v in srl_best_result.items()})
    
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

    
    



