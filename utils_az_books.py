import os, pathlib, sys
from tqdm import tqdm
import json
import torch
import torch.nn as nn
from typing import Any
import argparse

MAX_VAL = 1e4

def json_load(path: str):
    return json.load(open(path, "r"))

def json_dump(path: str, obj: Any):
    json.dump(obj, open(path, "w"))


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)


def min_occurrences(tensor):
    unique_elements, counts = torch.unique(tensor, return_counts=True)
    min_count = counts.min().item()
    num_min_elements = (counts == min_count).sum().item()
    return num_min_elements


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class Ranker(nn.Module):
    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, scores, labels):
        labels = labels.squeeze()
        
        try:
            loss = self.ce(scores, labels).item()
        except:
            print(scores.size())
            print(labels.size())
            loss = 0.0
        
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        
        valid_length = (scores > -MAX_VAL).sum(-1).float()
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            )
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        res.append((1 - (rank/valid_length)).mean().item()) # AUC
        return res


def e_str(e: tuple) -> str:
    return "-".join(e)


def get_metric():
    # dataset = sys.argv[1]
    match_model = sys.argv[1]
    seed = sys.argv[2]

    metrics = []
    r = f"./outputs_25m/{match_model}/{seed}/"
    for x in tqdm(os.listdir(r)):
        if "_" in x:
        # if True:
            file_path = os.path.join(r, x, "result.log")
            if os.path.exists(file_path):
                try:
                    lines = open(file_path, "r").readlines()
                    metric_dict = eval(lines[-1])
                    metrics.append((metric_dict["NDCG@10"], metric_dict["Recall@10"], metric_dict["MRR"], x))
                except:
                    pass
    metrics = sorted(metrics, key=lambda x:-x[0])
    print(len(metrics))
    for x in metrics[:30]:
        print(f"{float(x[0]):.5f}, {float(x[1]):.5f}, {float(x[2]):.5f}, {x[3]}")



def parse_args():
    parser = argparse.ArgumentParser()
    # Setup args
    parser.add_argument("--data_dir", type=str, default="./data/az-books/proc_data")
    parser.add_argument("--dataset", type=str, default="az-books")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric_ks", nargs='+', type=int, default=[1, 5, 10, 20, 30, 50])
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Training args
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_sched", type=str, default="constant")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Ablation args
    parser.add_argument("--num_codebook_used", type=int, default=-1)
    parser.add_argument("--not_on_user", action="store_true")
    parser.add_argument("--not_on_item", action="store_true")
    parser.add_argument("--remove_ui", action="store_true")
    parser.add_argument("--remove_iu", action="store_true")

    # Model args
    parser.add_argument("--model_name", type=str, default="DSSM")
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--num_interest", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--plus_llm_embed", action="store_true")
    parser.add_argument("--num_neg_items", type=int, default=50)
    parser.add_argument("--plus_quant_idx", action="store_true")
    parser.add_argument("--plus_on_hist", action="store_true")
    parser.add_argument("--plus_gnn_embed", action="store_true")
    parser.add_argument("--gnn_hidden_size", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--degree", type=int, default=15)
    parser.add_argument("--idx_type", type=str, default="quant")
    parser.add_argument("--num_quantizers", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--cold_start", action="store_true")
    parser.add_argument("--test_on_cold_u", action="store_true")
    parser.add_argument("--test_on_cold_i", action="store_true")
    parser.add_argument("--gnn_dropout", type=float, default=0.1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    get_metric()
