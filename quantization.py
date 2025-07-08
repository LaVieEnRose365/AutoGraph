import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from vector_quantize_pytorch import ResidualVQ

import os
import numpy as np
from transformers import set_seed, HfArgumentParser
from utils import json_load, weight_init
from tqdm import trange
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple


@dataclass
class TrainingArgs:
    lr: float = 1e-3
    wd: float = 1e-3
    epochs: int = 1000
    batch_size: int = 512
    seed: int = 42


@dataclass
class QuantArgs:
    dim: int = 32
    num_quantizers: int = 3
    codebook_size: int = 256
    kmeans_init: bool = True
    kmeans_iters: int = 500
    sample_codes: bool = False
    alpha: float = 0.25
    embed_dir: str = "outputs_llm"
    output_dir: str = "outputs_quant"
    embed_type: str = "item"
    dataset: str = "ml-1m"
    dim_dnn: Tuple = tuple([256, 128])

    def __post_init__(self):
        self.embed_path = f"{self.embed_dir}/{self.dataset}/vicuna-7b-v1.5_{self.embed_type}.npy"
        embed_dict = json_load(self.embed_path)
        meta_data = json_load("data/ml-1m/proc_data/match-meta.json")
        feature_dict = meta_data["feature_dict"]
        dict_field = "User ID" if self.embed_type == "user" else "Movie ID"
        matchID_to_id = {matchId: ori_id for ori_id, matchId in feature_dict[dict_field].items()}
        self.embeddings = np.array([embed_dict[matchID_to_id[i]] for i in range(len(matchID_to_id))])



class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        return torch.tensor(self.embeddings[index]).to(torch.float32)



class RQAutoEncoder(nn.Module):
    def __init__(self, config: QuantArgs):
        super().__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        for in_dim, out_dim in zip(config.dim_dnn[:-1], config.dim_dnn[1:]):
            self.encoder.append(nn.Linear(in_dim, out_dim))
            self.encoder.append(nn.ReLU())
        self.encoder.pop(-1)

        for in_dim, out_dim in zip(config.dim_dnn[::-1][:-1], config.dim_dnn[::-1][1:]):
            self.decoder.append(nn.Linear(in_dim, out_dim))
            self.decoder.append(nn.ReLU())
        self.decoder.pop(-1)
        
        self.rq_layer = ResidualVQ(
            dim = config.dim,
            codebook_size = config.codebook_size,
            num_quantizers = config.num_quantizers,
            kmeans_init = config.kmeans_init,
            kmeans_iters = config.kmeans_iters,
            stochastic_sample_codes = config.sample_codes,
            sample_codebook_temp = 0.1
        )

        self.post_init()

    def forward(self, x):
        x = self.encoder(x)
        x, indices, commit_loss = self.rq_layer(x)
        x = self.decoder(x)
        return x, indices, commit_loss
    
    def post_init(self):
        self.encoder.apply(weight_init)
        self.decoder.apply(weight_init)




if __name__ == "__main__":
    parser = HfArgumentParser((QuantArgs, TrainingArgs))
    quant_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct data and dataloader
    print("Constructing dataset.")
    embed_dataset = EmbeddingDataset(quant_args.embeddings)
    train_dataloader = DataLoader(dataset=embed_dataset, batch_size=training_args.batch_size, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(dataset=embed_dataset, batch_size=training_args.batch_size, shuffle=False, num_workers=4)

    # Train
    print("Constructing model.")
    model = RQAutoEncoder(quant_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.lr, weight_decay=training_args.wd)

    print("Training.")
    for epoch in (pbar := trange(training_args.epochs)):
        total_rec_loss = []
        total_cmt_loss = []
        total_indices = []
        model.train()
        for x in train_dataloader:
            x = x.to(device)
            x_rec, indices, cmt_loss = model(x)
            rec_loss = (x_rec - x).abs().mean()
            cmt_loss = cmt_loss.sum()
            loss = rec_loss + quant_args.alpha * cmt_loss
            
            total_rec_loss.append(rec_loss.item())
            total_cmt_loss.append(cmt_loss.item())
            total_indices.append(indices.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_indices = torch.cat(total_indices, dim=0).t()

            active_ratio = []
            active_min_count = []
            active_max_count = []

            for c, idx in enumerate(total_indices):
                uniques, counts = torch.unique(idx, return_counts=True)

                min_count = counts.min().item()
                max_count = counts.max().item()
                active_ratio.append(uniques.numel()/quant_args.codebook_size)
                active_min_count.append(min_count)
                active_max_count.append(max_count)
            assert len(active_ratio) == quant_args.num_quantizers                        

            pbar.set_description(f"Epoch {epoch}")
            pbar.set_postfix({
                "rec loss": f"{np.mean(total_rec_loss):.4f}",
                "cmt loss": f"{np.mean(total_cmt_loss):.4f}",
                "active ratio": ", ".join(map(lambda x: f"{100*x:.3f}%", active_ratio)),
                "min": ", ".join(map(lambda x: f"{x}", active_min_count)),
                "max": ", ".join(map(lambda x: f"{x}", active_max_count))
            })
    
    # Save quantization indices and embeddings
    model.eval()
    rec_embeds, total_indices = [], []
    with torch.no_grad():
        for x in eval_dataloader:
            x = x.to(device)
            x_rec, indices, _ = model(x)
            rec_embeds.append(x_rec.detach().cpu().numpy())
            total_indices.append(indices.detach().cpu().numpy())
    
    rec_embeds = np.concatenate(rec_embeds, axis=0)
    total_indices = np.concatenate(total_indices, axis=0)
    codebooks =  model.rq_layer.codebooks.detach().cpu().numpy()

    print("rec_embeds:", rec_embeds.shape)
    print("total indices:", total_indices.shape)
    print("codebooks", codebooks.shape)

    sub_dir = f"{quant_args.output_dir}/{quant_args.dataset}/{quant_args.embed_type}"
    os.makedirs(sub_dir, exist_ok=True)
    np.save(f"{sub_dir}/rec_embeds_cbNum{quant_args.num_quantizers}_cbSize{quant_args.codebook_size}.npy", rec_embeds)
    np.save(f"{sub_dir}/indices_cbNum{quant_args.num_quantizers}_cbSize{quant_args.codebook_size}.npy", total_indices)
    np.save(f"{sub_dir}/codebooks_cbNum{quant_args.num_quantizers}_cbSize{quant_args.codebook_size}.npy", codebooks)

    torch.save(model.state_dict(), f"{quant_args.output_dir}/{quant_args.embed_type}.pth")

