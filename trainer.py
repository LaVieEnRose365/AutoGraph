import logging
import os, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import pandas as pd
from tqdm import tqdm, trange
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import time
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from utils_az_books import Ranker, AverageMeterSet

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self, 
            model, 
            args,
            train_dataset: Dataset, 
            eval_dataset: Dataset, 
            data_collator_fn=None,
        ):
        self.model = model
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.n_gpu = self.args.n_gpu
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator_fn = data_collator_fn

        self.global_step = 0
        logger.info(f"setting device {self.device}")
        self.optimizer = None
        self.scheduler = None
        self.best_valid_ndcg = 0
        self.best_valid_step = -1


    def get_dataloader(self, dataset, is_training=True):
        collator = self.data_collator_fn
        # if not is_training and self.args.plus_lightgcn_embed:
        #     collator = data_collator_ccgnn

        dataloader = DataLoader(
            dataset, 
            batch_size=32 if not is_training else self.args.batch_size, 
            shuffle=is_training,
            num_workers=16,
            # collate_fn=self.data_collator_fn,
            collate_fn=collator,
            pin_memory=True
        )
        return dataloader

    def get_optimizer(self, num_training_steps: int, num_warmup_steps: int):
        no_decay = ["bias", "LayerNorm.weight"]
        named_params = [(k, v) for k, v in self.model.named_parameters()]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=self.args.lr, 
        )
        if self.args.lr_sched.lower() == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=num_training_steps, 
            )
        elif self.args.lr_sched.lower() == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=num_training_steps, 
            )
        elif self.args.lr_sched.lower() in ["const", "constant"]:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps, 
            )
        else:
            raise NotImplementedError(f"Unsupported lr schedule: {self.args.lr_sched}")

        return optimizer, scheduler

    def train(self):
        train_dataloader = self.get_dataloader(self.train_dataset)

        t_total = int(len(train_dataloader) * self.args.epochs)
        t_warmup = int(t_total * self.args.warmup_ratio)
        self.optimizer, self.scheduler = self.get_optimizer(num_training_steps=t_total, num_warmup_steps=t_warmup)

        logger.info("***** running training *****")
        logger.info(f"  model_name = {self.args.model_name}")
        logger.info(f"  dataset = {self.args.dataset}")
        logger.info(f"  num_user_feats = {self.args.num_user_feats}")
        logger.info(f"  num_item_feats = {self.args.num_item_feats}")
        logger.info(f"  num_user_fields = {self.args.num_user_fields}")
        logger.info(f"  num_item_fields = {self.args.num_item_fields}")
        logger.info(f"  num_examples = {len(self.train_dataset)}")
        logger.info(f"  num_epochs = {self.args.epochs}")
        logger.info(f"  batch_size = {self.args.batch_size}")
        logger.info(f"  total_steps = {t_total}")
        logger.info(f"  warmup_steps = {t_warmup}")
        logger.info(f"  learning_rate = {self.args.lr}")
        logger.info(f"  weight_decay = {self.args.weight_decay}")
        logger.info(f"  lr_sched = {self.args.lr_sched}")

        self._patience = 0
        self._stop_training = False
        self.global_step = 0

        self.model.to(self.device)
        self.model.zero_grad()
        loss_fct = nn.CrossEntropyLoss()

        with trange(self.args.epochs, desc="epoch", dynamic_ncols=True) as pbar:
            for epoch in pbar:
                logger.info(f"-------------------- epoch-{epoch} --------------------")
                self.model.train()
                epoch_loss = []
                for inputs in tqdm(train_dataloader):
                    for k in inputs:
                        inputs[k] = inputs[k].to(self.device)
                    outputs = self.model(**inputs)
                    logits, labels = outputs["logits"], outputs["labels"].to(self.device)
                    loss = loss_fct(logits, labels)
                    loss.backward()
                    step_loss = loss.item()
                    epoch_loss.append(step_loss)
                    pbar.set_description(f"epoch-{epoch}, loss={step_loss:.4f}")

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1
                logger.info(np.mean(epoch_loss))
                    
                self.evaluate()
                if self._stop_training:
                    break

    def evaluate(self, eval_dataset=None, descriptor="valid"):
        assert descriptor in ["valid", "test"]
        if eval_dataset is None:
            eval_dataloader = self.get_dataloader(self.eval_dataset, is_training=False)
        else:
            eval_dataloader = self.get_dataloader(eval_dataset, is_training=False)
        logger.info(f"***** running {descriptor} *****")
        logger.info(f"  num examples = {len(eval_dataloader.dataset)}")
        self.model.eval()

        ranker = Ranker(self.args.metric_ks)

        with torch.no_grad():
            all_logits, all_labels = [], []

            for inputs in tqdm(eval_dataloader):
                for k in inputs:
                    inputs[k] = inputs[k].to(self.device)
                outputs = self.model(**inputs)
                logits, labels = outputs["logits"], outputs["labels"]

                all_logits.append(logits.cpu().detach())
                all_labels.append(labels.cpu().detach())

            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels)
            res = ranker(all_logits, all_labels)

            metrics = {}
            for i, k in enumerate(self.args.metric_ks):
                metrics["NDCG@%d" % k] = res[2*i]
                metrics["Recall@%d" % k] = res[2*i+1]
            metrics["MRR"] = res[-2]
            metrics["AUC"] = res[-1]
            logger.info(metrics)

        if descriptor == "valid":
            if metrics["NDCG@10"] > self.best_valid_ndcg:
                self.best_valid_ndcg = metrics["NDCG@10"]
                self.best_valid_step = self.global_step
                self._patience = 0
                self.save_model(self.args.output_dir)
            else:
                self._patience += 1
            if self._patience > self.args.patience:
                self._stop_training = True

    def save_model(self, model_dir):
        save_dict = self.model.state_dict()
        torch.save(save_dict, os.path.join(model_dir, f"model.pt"))

    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, f"model.pt")
        # if "cuda" in self.device.type:
        #     state_dict = torch.load(model_path)
        # else:
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)

    def test(self, test_dataset: Dataset, model_dir=None):
        self.model.to(self.device)
        if model_dir is None:
            model_dir = self.args.output_dir
        self.load_model(model_dir)
        self.evaluate(test_dataset, "test")
