import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from dataclasses import dataclass
from layers import Squash, CapsuleLayer, Embeddings, TransformerEncoder
import numpy as np
import dgl
import dgl.nn as dglnn
from dgl.data.utils import load_graphs
from utils import weight_init


class MatchModel(nn.Module):
    used_params = ["embed_size"]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = Embeddings(config)
        item_table = torch.tensor(config.item_feats_table)
        self.item_feats_table = nn.Parameter(item_table, requires_grad=False)
        user_dim_in = config.num_user_fields*config.embed_size
        item_dim_in = config.num_item_fields*config.embed_size
        self.user_mlp = nn.Sequential(
            nn.Linear(user_dim_in, user_dim_in // 2),
            nn.ReLU(),
            nn.Linear(user_dim_in // 2, user_dim_in // 4),
            nn.ReLU(),
            nn.Linear(user_dim_in // 4, config.embed_size)
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(item_dim_in, item_dim_in // 2),
            nn.ReLU(),
            nn.Linear(item_dim_in // 2, config.embed_size),
        )

        if self.config.plus_gnn_embed:
            self.user_embed = nn.Embedding(config.num_users + 1, config.embed_size, padding_idx=config.num_users)
            self.item_embed = nn.Embedding(config.num_items, config.embed_size)
            self.user_quant_embed = nn.Embedding(config.u_code_num + 1, config.embed_size, padding_idx=config.u_code_num)
            self.item_quant_embed = nn.Embedding(config.i_code_num, config.embed_size)

            self.item_embed_i = nn.Embedding(config.num_items, config.embed_size)
            self.item_quant_embed_i = nn.Embedding(config.i_code_num, config.embed_size)
            self.user_embed_i = nn.Embedding(config.num_users + 1, config.embed_size, padding_idx=config.num_users)
            self.user_quant_embed_i = nn.Embedding(config.u_code_num + 1, config.embed_size, padding_idx=config.u_code_num)

            self.gnn = {
                ("user", "shares", "u_sem"): 
                    dglnn.GATConv(
                        (config.embed_size, config.embed_size),
                        config.gnn_hidden_size, 
                        num_heads=config.num_heads, 
                        allow_zero_in_degree=True,
                    ),

                ("item", "shares", "i_sem"): 
                    dglnn.GATConv(
                        (config.embed_size, config.embed_size),
                        config.gnn_hidden_size, 
                        num_heads=config.num_heads, 
                        allow_zero_in_degree=True,
                    ),

                ("i_sem", "denotes", "item"): 
                    dglnn.GATConv(
                        (config.gnn_hidden_size * config.num_heads + config.embed_size, config.embed_size), 
                        config.gnn_hidden_size, 
                        num_heads=config.num_heads, 
                        allow_zero_in_degree=True,
                    ),

                ("u_sem", "denotes", "user"): 
                    dglnn.GATConv(
                        (config.gnn_hidden_size * config.num_heads + config.embed_size, config.embed_size), 
                        config.gnn_hidden_size, 
                        num_heads=config.num_heads, 
                        allow_zero_in_degree=True,
                    ),

                ("item", "clicked_by", "user"): 
                    dglnn.GATConv(
                        (config.gnn_hidden_size * config.num_heads, config.gnn_hidden_size * config.num_heads), 
                        config.embed_size, 
                        num_heads=config.num_heads,
                        allow_zero_in_degree=True,
                    )
            }


            self.i_gnn = {
                ("user", "shares", "u_sem"): 
                    dglnn.GATConv(
                        (config.embed_size, config.embed_size),
                        config.gnn_hidden_size, 
                        num_heads=config.num_heads, 
                        allow_zero_in_degree=True,
                    ),

                ("item", "shares", "i_sem"): 
                    dglnn.GATConv(
                        (config.embed_size, config.embed_size),
                        config.gnn_hidden_size, 
                        num_heads=config.num_heads, 
                        allow_zero_in_degree=True,
                    ),

                ("i_sem", "denotes", "item"): 
                    dglnn.GATConv(
                        (config.gnn_hidden_size * config.num_heads + config.embed_size, config.embed_size), 
                        config.gnn_hidden_size, 
                        num_heads=config.num_heads, 
                        allow_zero_in_degree=True,
                    ),

                ("u_sem", "denotes", "user"): 
                    dglnn.GATConv(
                        (config.gnn_hidden_size * config.num_heads + config.embed_size, config.embed_size), 
                        config.gnn_hidden_size, 
                        num_heads=config.num_heads, 
                        allow_zero_in_degree=True,
                    ),

                ("user", "clicked", "item"): 
                    dglnn.GATConv(
                        (config.gnn_hidden_size * config.num_heads, config.gnn_hidden_size * config.num_heads), 
                        config.embed_size, 
                        num_heads=config.num_heads,
                        allow_zero_in_degree=True,
                    )
            }

            if self.config.remove_ui:
                mapping_size_ui = config.gnn_hidden_size * config.num_heads
            else:
                mapping_size_ui = config.embed_size * config.num_heads

            if self.config.remove_iu:
                mapping_size_iu = config.gnn_hidden_size * config.num_heads
            else:
                mapping_size_iu = config.embed_size * config.num_heads

            self.item_gnn_mapping = nn.Sequential(
                nn.Linear(mapping_size_iu, config.embed_size),
                nn.ReLU(),
                nn.Dropout(self.config.gnn_dropout)
            )

            self.user_gnn_mapping = nn.Sequential(
                nn.Linear(mapping_size_ui, config.embed_size),
                nn.ReLU(),
                nn.Dropout(self.config.gnn_dropout)
            )

            for relation in self.gnn:
                self.gnn[relation] = self.gnn[relation].cuda()
            
            for relation in self.i_gnn:
                self.i_gnn[relation] = self.i_gnn[relation].cuda()


    def u_graph_propagation(self, g):
        user_emb = self.user_embed(g.nodes["user"].data["id"])
        u_quant_emb = self.user_quant_embed(g.nodes["u_sem"].data["id"])

        # Aggregate user/item neighbors info to quant idx nodes.
        etype = ("user", "shares", "u_sem")
        u_n = self.gnn[etype](g.edge_type_subgraph([etype]), (user_emb, u_quant_emb)).flatten(-2)

        # Aggregate quant idx nodes info to user/item nodes.
        etype = ("u_sem", "denotes", "user")
        u_h = self.gnn[etype](g.edge_type_subgraph([etype]), (torch.cat([u_n, u_quant_emb], dim=-1), user_emb)).flatten(-2)
        
        if self.config.remove_ui:
            h = u_h
        else:
            item_emb = self.item_embed(g.nodes["item"].data["id"])
            i_quant_emb = self.item_quant_embed(g.nodes["i_sem"].data["id"])

            etype = ("item", "shares", "i_sem")
            i_n = self.gnn[etype](g.edge_type_subgraph([etype]), (item_emb, i_quant_emb)).flatten(-2)

            etype = ("i_sem", "denotes", "item")
            i_h = self.gnn[etype](g.edge_type_subgraph([etype]), (torch.cat([i_n, i_quant_emb], dim=-1), item_emb)).flatten(-2)

            # Aggregate item nodes info to user nodes.
            etype = ("item", "clicked_by", "user")
            h = self.gnn[etype](g.edge_type_subgraph([etype]), (i_h, u_h)).flatten(-2)

        # Select target users.
        h = h[g.nodes["user"].data["mask"].bool()]
        h = self.user_gnn_mapping(h)

        return h


    def i_graph_propagation(self, g):
        item_emb = self.item_embed_i(g.nodes["item"].data["id"])
        i_quant_emb = self.item_quant_embed_i(g.nodes["i_sem"].data["id"])

        etype = ("item", "shares", "i_sem")
        i_n = self.i_gnn[etype](g.edge_type_subgraph([etype]), (item_emb, i_quant_emb)).flatten(-2)


        etype = ("i_sem", "denotes", "item")
        i_h = self.i_gnn[etype](g.edge_type_subgraph([etype]), (torch.cat([i_n, i_quant_emb], dim=-1), item_emb)).flatten(-2)

        if self.config.remove_iu:
            h = i_h
        else:
            user_emb = self.user_embed_i(g.nodes["user"].data["id"])
            u_quant_emb = self.user_quant_embed_i(g.nodes["u_sem"].data["id"])

            # Aggregate user/item neighbors info to quant idx nodes.
            etype = ("user", "shares", "u_sem")
            u_n = self.i_gnn[etype](g.edge_type_subgraph([etype]), (user_emb, u_quant_emb)).flatten(-2)

            # Aggregate quant idx nodes info to user/item nodes.
            etype = ("u_sem", "denotes", "user")
            u_h = self.i_gnn[etype](g.edge_type_subgraph([etype]), (torch.cat([u_n, u_quant_emb], dim=-1), user_emb)).flatten(-2)
            
            # Aggregate user nodes info to item nodes.
            etype = ("user", "clicked", "item")
            h = self.i_gnn[etype](g.edge_type_subgraph([etype]), (u_h, i_h)).flatten(-2)

        # Select target items.
        h = h[g.nodes["item"].data["mask"].bool()]
        h = self.item_gnn_mapping(h)
        return h
    

    def item_tower(self, pos_item_ids, neg_item_ids, **kwargs):
        """Get item ID embeddings."""

        bs = pos_item_ids.shape[0]
        if self.training:
            item_ids = torch.cat([pos_item_ids.unsqueeze(1), neg_item_ids], dim=1)
            item_feats = self.item_feats_table[item_ids]
            emb = self.embed({"item_feats": item_feats})["item_feats"] # (bs, cnt, F, D)

            if self.config.plus_gnn_embed and not self.config.not_on_item:
                gnn_emb = self.i_graph_propagation(kwargs["i_graphs"])
                gnn_emb = gnn_emb.reshape(bs, item_ids.shape[1], self.config.embed_size)
                emb = torch.cat([emb, gnn_emb.unsqueeze(2)], dim=2)

            item_embed = self.item_mlp(emb.flatten(2))
            labels = torch.zeros(bs)
            return item_embed, labels
        else:
            item_feats_embed = self.embed({"item_feats": self.item_feats_table})["item_feats"] # (num_items, F, D)
            item_feats_embed = item_feats_embed.unsqueeze(0)

            if self.config.plus_gnn_embed and not self.config.not_on_item:
                gnn_emb = self.i_graph_propagation(self.config.i_graphs).unsqueeze(0)
                item_feats_embed = torch.cat([item_feats_embed, gnn_emb.unsqueeze(2)], dim=2)

            item_embed_pool = self.item_mlp(item_feats_embed.flatten(2))
            labels = pos_item_ids
            return item_embed_pool, labels


    def user_tower(self, user_feats, hist_ids, hist_mask, **kwargs):
        """Get user embeddings."""

        embeds = self.embed({
            "user_feats": user_feats,
            "item_id_u_tower": hist_ids
        })

        hist_embed  = embeds["item_id_u_tower"]
        user_embed = embeds["user_feats"]

        if self.config.plus_gnn_embed and not self.config.not_on_user:
            gnn_emb = self.u_graph_propagation(kwargs["u_graphs"]).unsqueeze(1)
            user_embed = torch.cat([user_embed, gnn_emb], dim=1)

        user_embed = self.hist_aggregation(user_embed, hist_embed, hist_mask)
        return user_embed


    def hist_aggregation(self, user_embed, hist_embed, hist_mask):
        raise NotImplementedError


    def forward(self, user_feats, hist_ids, hist_mask, pos_item_ids, neg_item_ids, **kwargs):
        user_embed = self.user_tower(user_feats, hist_ids, hist_mask, **kwargs)
        item_embed, labels = self.item_tower(pos_item_ids, neg_item_ids, **kwargs)
        logits = torch.sum(user_embed.unsqueeze(1) * item_embed, dim=2)

        outputs = {
            "logits": logits,
            "labels": labels.long()
        }

        return outputs


class DSSM(MatchModel):
    def __init__(self, config):
        super().__init__(config)
    
    def hist_aggregation(self, user_embed, hist_embed, hist_mask):
        hist_embed = torch.sum(hist_embed * hist_mask.unsqueeze(-1), dim=1) / torch.sum(hist_mask, dim=1, keepdim=True)
        user_embed = torch.cat([user_embed, hist_embed.unsqueeze(1)], dim=1)
        user_embed = self.user_mlp(user_embed.flatten(1))
        return user_embed


class MIND(MatchModel):
    def __init__(self, config):
        super().__init__(config)
        self.capsule_layer = CapsuleLayer(config)

    def hist_aggregation(self, user_embed, hist_embed, hist_mask):
        caps = self.capsule_layer(hist_embed, hist_mask)
        user_embed = torch.cat([user_embed, caps], dim=1)
        user_embed = self.user_mlp(user_embed.flatten(1))
        return user_embed


class GRU4Rec(MatchModel):
    def __init__(self, config):
        super().__init__(config)
        self.gru = nn.GRU(input_size=config.embed_size,
                          hidden_size=config.embed_size,
                          num_layers=config.num_layers,
                          batch_first=True)
        

    def hist_aggregation(self, user_embed, hist_embed, hist_mask):
        hist_lens = hist_mask.sum(dim=1).cpu()
        packed_seq = pack_padded_sequence(hist_embed, hist_lens, batch_first=True, enforce_sorted=False)
        _, final_hidden = self.gru(packed_seq)
        hist_embed = final_hidden[-1,:,:]

        user_embed = torch.cat([user_embed, hist_embed.unsqueeze(1)], dim=1)
        user_embed = self.user_mlp(user_embed.flatten(1))

        return user_embed


class SASRec(MatchModel):
    def __init__(self, config):
        super().__init__(config)
        self.position_embed = nn.Embedding(100, config.embed_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=config.num_layers,
            n_heads=config.num_heads,
            hidden_size=config.embed_size,
            inner_size=4*config.embed_size,
        )
        self.LayerNorm = nn.LayerNorm(config.embed_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)


    def hist_aggregation(self, user_embed, hist_embed, hist_mask):
        position_ids = torch.arange(
            hist_embed.shape[1], dtype=torch.long, device=hist_embed.device
        )
        position_ids = position_ids.unsqueeze(0).expand(hist_embed.shape[0], -1)
        position_embedding = self.position_embed(position_ids)

        hist_embed += position_embedding
        hist_embed = self.LayerNorm(hist_embed)
        hist_embed = self.dropout(hist_embed)

        # Causal Mask
        attention_mask = hist_mask.bool()
        extended_att_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_att_mask = torch.tril(
            extended_att_mask.expand((-1, -1, hist_embed.shape[1], -1))
        )
        
        extended_att_mask = torch.where(extended_att_mask, 0.0, -10000.0)

        trm_output = self.trm_encoder(
            hist_embed, extended_att_mask
        )[-1]

        # Gather index
        gather_index = torch.sum(hist_mask, dim=-1) - 1
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, trm_output.shape[-1])
        output_tensor = trm_output.gather(dim=1, index=gather_index) # (B, 1, H)

        user_embed = torch.cat([user_embed, output_tensor], dim=1)
        user_embed = self.user_mlp(user_embed.flatten(1))

        return user_embed
        


class SDM(MatchModel):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(config.embed_size, config.embed_size, num_layers=2, batch_first=True, dropout=0.1)
        self.multihead_att = nn.MultiheadAttention(config.embed_size, 2, dropout=0.1, batch_first=True)
        self.gating = nn.Linear(3*config.embed_size, config.embed_size)

        user_dim_in = (config.num_user_fields - 1)*config.embed_size
        self.map_fc = nn.Linear(user_dim_in, config.embed_size)

    def hist_aggregation(self, user_embed, hist_embed, hist_mask):
        mapped_user_embed = self.map_fc(user_embed.flatten(1))

        # Short term
        lstm_embed, (_, _) = self.lstm(hist_embed[:, -10:])
        self_att_embed, _ = self.multihead_att(lstm_embed, lstm_embed, lstm_embed, need_weights=False)
        short_att_score = torch.einsum("bld,bd->bl", self_att_embed, mapped_user_embed).softmax(dim=1)
        short_term_embed = torch.einsum("bl,bd->bd", short_att_score, mapped_user_embed)

        # Long term
        hist_mask = (1 - hist_mask).bool()
        long_att_score = torch.einsum("bld,bd->bl", hist_embed[:, :20], mapped_user_embed)
        long_att_score[hist_mask[:, :20]] = -1e9
        long_att_score = long_att_score.softmax(dim=1)
        long_term_embed = torch.einsum("bl,bd->bd", long_att_score, mapped_user_embed)

        gating_score = self.gating(torch.cat([short_term_embed, long_term_embed, mapped_user_embed], dim=1)).sigmoid()
        hist_embed = (1 - gating_score) * short_term_embed + gating_score * long_term_embed

        user_embed = torch.cat([user_embed, hist_embed.unsqueeze(1)], dim=1)
        user_embed = self.user_mlp(user_embed.flatten(1))

        return user_embed
