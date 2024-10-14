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
from utils_az_books import weight_init


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

        if self.config.plus_llm_embed:
            self.llm_embeds = nn.ModuleDict({
                "user": nn.Embedding.from_pretrained(config.llm_embeds["user"]),
                "item": nn.Embedding.from_pretrained(config.llm_embeds["item"])
            })

            self.user_llm_mlp = nn.Linear(config.llm_embed_dim, config.embed_size)
            self.item_llm_mlp = nn.Linear(config.llm_embed_dim, config.embed_size)

        if self.config.plus_lightgcn_embed:
            self.lightgcn_embed = nn.ModuleDict({
                # "user_u": nn.Embedding(config.num_users, config.embed_size),
                # "item_u": nn.Embedding(config.num_items, config.embed_size),
                "user": nn.Embedding(config.num_users + 1, config.embed_size, padding_idx=config.num_users),
                "item": nn.Embedding(config.num_items, config.embed_size)
            })

            self.lightgcn_u = dglnn.GraphConv(config.embed_size, config.embed_size, weight=True, bias=True, allow_zero_in_degree=True)
            self.lightgcn_i = dglnn.GraphConv(config.embed_size, config.embed_size, weight=True, bias=True, allow_zero_in_degree=True)

        if self.config.plus_topoI2I_embed:
            self.topoI2I_embed = nn.ModuleDict({
                "user_u": nn.Embedding(config.num_users, config.embed_size),
                "item_u": nn.Embedding(config.num_items, config.embed_size),
                "item_i": nn.Embedding(config.num_items, config.embed_size)
            })

            self.topoI2I_i = dglnn.GATConv(config.embed_size, config.embed_size, num_heads=config.num_heads, allow_zero_in_degree=True)
            self.topoI2I_u = {
                ("item", "similar_to", "item"):
                    dglnn.GATConv(
                        (config.embed_size, config.embed_size),
                        config.gnn_hidden_size,
                        num_heads=config.num_heads,
                        allow_zero_in_degree=True
                    ),
                ("item", "clicked_by", "user"):
                    dglnn.GATConv(
                        (config.gnn_hidden_size * config.num_heads + config.embed_size, config.embed_size),
                        config.embed_size,
                        num_heads=config.num_heads,
                        allow_zero_in_degree=True
                    )
            }

            for relation in self.topoI2I_u:
                self.topoI2I_u[relation] = self.topoI2I_u[relation].cuda()

            self.topoI2I_mlp = nn.ModuleDict({
                "user": nn.Sequential(
                    nn.Linear(config.embed_size * config.num_heads, config.embed_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ),
                "item": nn.Sequential(
                    nn.Linear(config.embed_size * config.num_heads, config.embed_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            })

        if self.config.plus_ccgnn_embed:
            self.ccgnn_embed = nn.ModuleDict({
                "user": nn.Embedding(config.num_users, config.embed_size),
                "item_u": nn.Embedding(config.num_items, config.embed_size),
                "item_i": nn.Embedding(config.num_items, config.embed_size),
                "genre_u": nn.Embedding(config.num_genres, config.embed_size),
                "genre_i": nn.Embedding(config.num_genres, config.embed_size)
            })

            self.ccgnn_u = {
                ("item", "belongs_to", "genre"): 
                    dglnn.GATConv(
                        (config.embed_size, config.embed_size),
                        config.gnn_hidden_size,
                        num_heads=config.num_heads,
                        allow_zero_in_degree=True
                    ),

                ("genre", "contains", "item"): 
                    dglnn.GATConv(
                        (config.gnn_hidden_size * config.num_heads + config.embed_size, config.gnn_hidden_size),
                        config.gnn_hidden_size,
                        num_heads=config.num_heads,
                        allow_zero_in_degree=True
                    ),

                ("item", "clicked_by", "user"):
                    dglnn.GATConv(
                        (config.gnn_hidden_size * config.num_heads + config.embed_size, config.embed_size),
                        config.embed_size,
                        num_heads=config.num_heads,
                        allow_zero_in_degree=True
                    )
            }

            self.ccgnn_i = {
                ("item", "belongs_to", "genre"):
                    dglnn.GATConv(
                        (config.embed_size, config.embed_size),
                        config.gnn_hidden_size,
                        num_heads=config.num_heads,
                        allow_zero_in_degree=True
                    ),
                ("genre", "contains", "item"):
                    dglnn.GATConv(
                        (config.gnn_hidden_size * config.num_heads + config.embed_size, config.gnn_hidden_size),
                        config.embed_size,
                        num_heads=config.num_heads,
                        allow_zero_in_degree=True
                    )
            }

            self.ccgnn_mlp = nn.ModuleDict({
                "user": nn.Sequential(
                    nn.Linear(config.embed_size * config.num_heads, config.embed_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ),
                "item": nn.Sequential(
                    nn.Linear(config.embed_size * config.num_heads, config.embed_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            })

            
            for relation in self.ccgnn_u:
                self.ccgnn_u[relation] = self.ccgnn_u[relation].cuda()
            
            for relation in self.ccgnn_i:
                self.ccgnn_i[relation] = self.ccgnn_i[relation].cuda()


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


    def ccgnn_i_graph_propagation(self, g):
        item_emb = self.ccgnn_embed["item_i"](g.nodes["item"].data["id"])
        genre_emb = self.ccgnn_embed["genre_i"](g.nodes["genre"].data["id"])

        etype = ("item", "belongs_to", "genre")
        i_n = self.ccgnn_i[etype](g.edge_type_subgraph([etype]), (item_emb, genre_emb)).flatten(-2)

        etype = ("genre", "contains", "item")
        h = self.ccgnn_i[etype](g.edge_type_subgraph([etype]), (torch.cat([i_n, genre_emb], dim=-1), item_emb)).flatten(-2)

        h = self.ccgnn_mlp["item"](h)

        return h


    def ccgnn_u_graph_propagation(self, g):
        user_emb = self.ccgnn_embed["user"](g.nodes["user"].data["id"])
        item_emb = self.ccgnn_embed["item_u"](g.nodes["item"].data["id"])
        genre_emb = self.ccgnn_embed["genre_u"](g.nodes["genre"].data["id"])

        etype = ("item", "belongs_to", "genre")
        i_n = self.ccgnn_u[etype](g.edge_type_subgraph([etype]), (item_emb, genre_emb)).flatten(-2)

        etype = ("genre", "contains", "item")
        h = self.ccgnn_u[etype](g.edge_type_subgraph([etype]), (torch.cat([i_n, genre_emb], dim=-1), item_emb)).flatten(-2)

        etype = ("item", "clicked_by", "user")
        h = self.ccgnn_u[etype](g.edge_type_subgraph([etype]), (torch.cat([h, item_emb], dim=-1), user_emb)).flatten(-2)

        h = self.ccgnn_mlp["user"](h)

        return h


    def lightgcn_u_graph_propagation(self, g):
        user_emb = self.lightgcn_embed["user"](g.nodes["user"].data["id"])
        item_emb = self.lightgcn_embed["item"](g.nodes["item"].data["id"])

        h = self.lightgcn_u(g, (item_emb, user_emb))
        return h


    def lightgcn_i_graph_propagation(self, g):
        user_emb = self.lightgcn_embed["user"](g.nodes["user"].data["id"])
        item_emb = self.lightgcn_embed["item"](g.nodes["item"].data["id"])

        h = self.lightgcn_i(g, (user_emb, item_emb))
        return h


    def topoI2I_u_graph_propagation(self, g):
        user_emb = self.topoI2I_embed["user_u"](g.nodes["user"].data["id"])
        item_emb = self.topoI2I_embed["item_u"](g.nodes["item"].data["id"])

        etype = ("item", "similar_to", "item")
        i_n = self.topoI2I_u[etype](g.edge_type_subgraph([etype]), (item_emb, item_emb)).flatten(-2)

        etype = ("item", "clicked_by", "user")
        h = self.topoI2I_u[etype](g.edge_type_subgraph([etype]), (torch.cat([i_n, item_emb], dim=-1), user_emb)).flatten(-2)

        h = self.topoI2I_mlp["user"](h)

        return h

    def topoI2I_i_graph_propagation(self, g):
        item_emb = self.topoI2I_embed["item_i"].weight
        h = self.topoI2I_i(g, item_emb).flatten(-2)
        h = self.topoI2I_mlp["item"](h)
        return h


    def item_tower(self, pos_item_ids, neg_item_ids, **kwargs):
        """Get item ID embeddings."""

        bs = pos_item_ids.shape[0]
        if self.training:
        # if True:
            item_ids = torch.cat([pos_item_ids.unsqueeze(1), neg_item_ids], dim=1)
            item_feats = self.item_feats_table[item_ids]
            emb = self.embed({"item_feats": item_feats})["item_feats"] # (bs, cnt, F, D)
            if self.config.plus_llm_embed:
                llm_item_embed = self.llm_embeds["item"](item_ids)
                llm_item_embed = self.item_llm_mlp(llm_item_embed) # (bs, cnt, D)
                emb = torch.cat([emb, llm_item_embed.unsqueeze(2)], dim=2)

            if self.config.plus_gnn_embed and not self.config.not_on_item:
                gnn_emb = self.i_graph_propagation(kwargs["i_graphs"])
                gnn_emb = gnn_emb.reshape(bs, item_ids.shape[1], self.config.embed_size)
                emb = torch.cat([emb, gnn_emb.unsqueeze(2)], dim=2)

            if self.config.plus_lightgcn_embed:
                gnn_emb = self.lightgcn_i_graph_propagation(kwargs["i_graphs"])
                gnn_emb = gnn_emb.reshape(bs, item_ids.shape[1], self.config.embed_size)
                emb = torch.cat([emb, gnn_emb.unsqueeze(2)], dim=2)

            if self.config.plus_topoI2I_embed:
                gnn_emb = self.topoI2I_i_graph_propagation(kwargs["i_graphs"])[item_ids]
                emb = torch.cat([emb, gnn_emb.unsqueeze(2)], dim=2)

            if self.config.plus_ccgnn_embed:
                ccgnn_emb = self.ccgnn_i_graph_propagation(kwargs["i_graphs"])[item_ids]
                emb = torch.cat([emb, ccgnn_emb.unsqueeze(2)], dim=2)

            item_embed = self.item_mlp(emb.flatten(2))
            labels = torch.zeros(bs)
            return item_embed, labels
        else:
            item_feats_embed = self.embed({"item_feats": self.item_feats_table})["item_feats"] # (num_items, F, D)
            item_feats_embed = item_feats_embed.unsqueeze(0)
            if self.config.plus_llm_embed:
                llm_item_embed = self.item_llm_mlp(self.llm_embeds["item"].weight).unsqueeze(0)
                item_feats_embed = torch.cat([item_feats_embed, llm_item_embed.unsqueeze(2)], dim=2)

            if self.config.plus_gnn_embed and not self.config.not_on_item:
                gnn_emb = self.i_graph_propagation(self.config.i_graphs).unsqueeze(0)
                # gnn_emb = gnn_emb.reshape(bs, self.config.num_items, self.config.embed_size)                
                item_feats_embed = torch.cat([item_feats_embed, gnn_emb.unsqueeze(2)], dim=2)

            if self.config.plus_lightgcn_embed:
                gnn_emb = self.lightgcn_i_graph_propagation(self.config.i_graphs).unsqueeze(0)
                item_feats_embed = torch.cat([item_feats_embed, gnn_emb.unsqueeze(2)], dim=2)

            if self.config.plus_ccgnn_embed:
                ccgnn_emb = self.ccgnn_i_graph_propagation(kwargs["i_graphs"]).unsqueeze(0)
                item_feats_embed = torch.cat([item_feats_embed, ccgnn_emb.unsqueeze(2)], dim=2)

            if self.config.plus_topoI2I_embed:
                topoI2I_emb = self.topoI2I_i_graph_propagation(kwargs["i_graphs"]).unsqueeze(0)
                item_feats_embed = torch.cat([item_feats_embed, topoI2I_emb.unsqueeze(2)], dim=2)

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
        
        if self.config.plus_llm_embed:
            llm_user_embed = self.llm_embeds["user"](user_feats[:, 0])
            llm_user_embed = self.user_llm_mlp(llm_user_embed)
            user_embed = torch.cat([user_embed, llm_user_embed.unsqueeze(1)], dim=1)

            if self.config.plus_on_hist:
                llm_item_embed = self.llm_embeds["item"](hist_ids) # (bs, L, D)
                llm_item_embed = self.item_llm_mlp(llm_item_embed) # (bs, L, D)
                llm_item_embed = torch.sum(llm_item_embed * hist_mask.unsqueeze(-1), dim=1) / torch.sum(hist_mask, dim=1, keepdim=True) # (bs, D)
                user_embed = torch.cat([user_embed, llm_item_embed.unsqueeze(1)], dim=1)
        
        if self.config.plus_on_hist and self.config.plus_quant_idx:
            quant_item_idx = self.item_feats_table[hist_ids][:, :, -3:] # (bs, L, 3)
            quant_item_embed = self.embed({"item_feats": quant_item_idx})["item_feats"] # (bs, L, 3, D)
            quant_item_embed = torch.einsum("blnd,bl->blnd", quant_item_embed, hist_mask).sum(1)/ torch.sum(hist_mask, dim=1, keepdim=True).unsqueeze(-1)  
            user_embed = torch.cat([user_embed, quant_item_embed], dim=1)

        if self.config.plus_gnn_embed and not self.config.not_on_user:
            gnn_emb = self.u_graph_propagation(kwargs["u_graphs"]).unsqueeze(1)
            user_embed = torch.cat([user_embed, gnn_emb], dim=1)
        
        if self.config.plus_lightgcn_embed:
            gnn_emb = self.lightgcn_u_graph_propagation(kwargs["u_graphs"])
            user_embed = torch.cat([user_embed, gnn_emb.unsqueeze(1)], dim=1)

        if self.config.plus_ccgnn_embed:
            ccgnn_emb = self.ccgnn_u_graph_propagation(kwargs["u_graphs"]).unsqueeze(1)
            user_embed = torch.cat([user_embed, ccgnn_emb], dim=1)
        
        if self.config.plus_topoI2I_embed:
            topoI2I_emb = self.topoI2I_u_graph_propagation(kwargs["u_graphs"]).unsqueeze(1)
            user_embed = torch.cat([user_embed, topoI2I_emb], dim=1)

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