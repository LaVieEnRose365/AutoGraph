import json
import logging
import h5py
import os
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import dgl
from dgl.data.utils import load_graphs, save_graphs
from dgl.sampling import sample_neighbors
from utils import json_dump, json_load
import re

logger = logging.getLogger(__name__)


class MatchDataset:
    def __init__(self, args):
        self.args = args
        self.split_names = ["train", "valid", "test"]
        self.load_data()
    
    def load_data(self):
        meta_data = json.load(open(os.path.join(self.args.data_dir, "match-meta.json"), "r"))
        self.user_fields = meta_data["user_fields"]
        self.item_fields = meta_data["item_fields"]
        self.user_feat_count = meta_data["user_feat_count"]
        self.item_feat_count = meta_data["item_feat_count"]
        self.user_feat_offset = meta_data["user_feat_offset"]
        self.item_feat_offset = meta_data["item_feat_offset"]
        self.item_feats_table = torch.tensor(meta_data["item_feats_table"])
        self.feature_dict = meta_data["feature_dict"]
        self.item_to_users = meta_data["item_to_users"]

        self.num_user_feats = sum(self.user_feat_count)
        self.num_item_feats = sum(self.item_feat_count)
        self.num_items = self.item_feat_count[0]
        self.num_users = self.user_feat_count[0]
        self.num_item_fields = len(self.item_fields)
        self.num_user_fields = len(self.user_fields) + 1 # hist
        if self.args.model_name.lower() == "mind":
            self.num_user_fields += self.args.num_interest - 1
        

        user_offset = np.array(self.user_feat_offset).reshape(1, len(self.user_fields))
        item_offset = np.array(self.item_feat_offset).reshape(1, len(self.item_fields))

        with h5py.File(os.path.join(self.args.data_dir, f"match.h5"), "r") as f:
            self.user_data = {split: f[f"{split} user data"][:] + user_offset for split in self.split_names}
            self.item_data = {split: f[f"{split} item data"][:] + item_offset for split in self.split_names}
            self.hist_ids = {split: f[f"{split} history ID"][:] for split in self.split_names}
            self.hist_mask = {split: f[f"{split} history mask"][:] for split in self.split_names}


        if self.args.plus_gnn_embed:
            self.num_user_fields += 1 if not self.args.not_on_user else 0
            self.num_item_fields += 1 if not self.args.not_on_item else 0

            u_codebook_num, i_codebook_num = map(int, self.args.num_quantizers.split("|"))
            u_codebook_size, i_codebook_size = map(int, self.args.codebook_size.split("|"))

            # User quant idx
            u_quant_idx = np.load(f"{self.args.quant_dir}/{self.args.dataset}/user/indices_cbNum{u_codebook_num}_cbSize{u_codebook_size}.npy")
            u_quant_idx += np.arange(u_codebook_num) * u_codebook_size

            u_codebook_used = u_codebook_num if self.args.num_codebook_used == -1 else self.args.num_codebook_used
            self.u_code_num = u_codebook_used * u_codebook_size
            self.u_quant_idx = u_quant_idx[:, :u_codebook_used]

            # Item quant idx
            i_quant_idx = np.load(f"{self.args.quant_dir}/{self.args.dataset}/item/indices_cbNum{i_codebook_num}_cbSize{i_codebook_size}.npy")
            i_quant_idx += np.arange(i_codebook_num) * i_codebook_size

            i_codebook_used = i_codebook_num if self.args.num_codebook_used == -1 else self.args.num_codebook_used
            self.i_code_num = i_codebook_used * i_codebook_size
            self.i_quant_idx = i_quant_idx[:, :i_codebook_used]



    def get_splited_dataset(self, split):
        assert split in self.split_names, f"Unsupported split name: {split}"
        if self.args.plus_gnn_embed:
            return GraphDataset(
                split,
                self.args,
                self.user_data[split],
                self.item_data[split][:, 0],
                self.hist_ids[split],
                self.hist_mask[split],
                self.num_items,
                self.args.num_neg_items,
                self.u_quant_idx,
                self.i_quant_idx,
                self.u_code_num,
                self.i_code_num,
                self.item_to_users
            )

        else:
            return BaseDataset(
                split,
                self.args,
                self.user_data[split],
                self.item_data[split][:, 0],
                self.hist_ids[split],
                self.hist_mask[split],
                self.num_items,
                self.args.num_neg_items,
            )



class BaseDataset(Dataset):
    def __init__(self, split, args, user_data, pos_item_ids, hist_ids, hist_mask, num_items, num_neg_items):
        super().__init__()
        self.split = split
        self.args = args
        self.user_data = user_data
        self.pos_item_ids = pos_item_ids
        self.hist_ids = hist_ids
        self.hist_mask = hist_mask
        self.num_items = num_items
        self.num_neg_items = num_neg_items
    

    def __len__(self):
        return len(self.user_data)
    
    def __getitem__(self, idx):
        return self.user_data[idx], self.hist_ids[idx], self.hist_mask[idx], self.pos_item_ids[idx], self.gen_neg_items(idx)

    def gen_neg_items(self, idx):
        pos_item_id = self.pos_item_ids[idx]
        neg_item_ids = []
        while len(neg_item_ids) < self.num_neg_items:
            neg_item_id = np.random.randint(0, self.num_items)
            if neg_item_id != pos_item_id and neg_item_id not in neg_item_ids:
                neg_item_ids.append(neg_item_id)
        return np.array(neg_item_ids)


class GraphDataset(BaseDataset):
    def __init__(self, split, args, user_data, pos_item_ids, hist_ids, hist_mask, num_items, num_neg_items,
                  u_quant_idx, i_quant_idx, u_code_num, i_code_num, item_to_users):
        super().__init__(split, args, user_data, pos_item_ids, hist_ids, hist_mask, num_items, num_neg_items)
        self.u_quant_idx = u_quant_idx
        self.i_quant_idx = i_quant_idx
        self.u_code_num = u_code_num
        self.i_code_num = i_code_num
        self.item_to_users = item_to_users

        self.u_code_neighbors = [[] for _ in range(self.u_code_num+1)]
        self.i_code_neighbors = [[] for _ in range(self.i_code_num)]

        for uid, quant_idx in enumerate(self.u_quant_idx):
            for idx in quant_idx:
                self.u_code_neighbors[idx].append(uid)

        for iid, quant_idx in enumerate(self.i_quant_idx):
            for idx in quant_idx:
                self.i_code_neighbors[idx].append(iid)

        self.prepare_i_graphs()


    def prepare_i_graphs(self):
        np.random.seed(42)
        self.visible_u_code_neighbors = [np.random.choice(i, min(len(i), self.args.degree), replace=False) for i in self.u_code_neighbors]
        self.visible_i_code_neighbors = [np.random.choice(i, min(len(i), self.args.degree), replace=False) for i in self.i_code_neighbors]
        if self.split == "train":
            self.i_graphs = [self.construct_i_graph(iid) for iid in trange(self.num_items, desc="Preparing item graphs: ")]
        

    def __getitem__(self, idx):
        uid = self.user_data[idx, 0]
        pos_id = self.pos_item_ids[idx]
        neg_ids = self.gen_neg_items(idx)
        if self.split == "train":
            i_graph = dgl.batch([self.remove_u(self.i_graphs[pos_id], uid)] + [self.i_graphs[neg_id] for neg_id in neg_ids])
        else:
            return self.user_data[idx], self.hist_ids[idx], self.hist_mask[idx], \
                    pos_id, neg_ids, self.construct_u_graph(idx)

        return self.user_data[idx], self.hist_ids[idx], self.hist_mask[idx], \
                pos_id, neg_ids, self.construct_u_graph(idx), i_graph


    def construct_u_graph(self, idx):
        uid = self.user_data[idx, 0]
        
        # remap user id
        u_dict = {uid: 0}
        u_quant_idx = self.u_quant_idx[uid]
        u_neighbors = [self.visible_u_code_neighbors[i] for i in u_quant_idx]
        for neighbors in u_neighbors:
            for n in neighbors:
                if n not in u_dict:
                    u_dict[n] = len(u_dict)
        
        u_dict_reverse = {v: k for k, v in u_dict.items()}

        src_u2s = [u_dict[n] for neighbors in u_neighbors for n in neighbors]
        dst_u2s = np.arange(len(u_quant_idx)).repeat([len(neighbors) for neighbors in u_neighbors])

        if self.args.remove_ui:
            edges_dict = {
                ("u_sem", "denotes", "user"): (
                    torch.arange(len(u_quant_idx)),
                    torch.tensor([0]*len(u_quant_idx)), 
                ),

                ("user", "shares", "u_sem"): (
                    torch.tensor(src_u2s),
                    torch.tensor(dst_u2s)
                )
            }

            g = dgl.heterograph(edges_dict)
            g.nodes["user"].data["id"] = torch.tensor([u_dict_reverse[i] for i in range(len(u_dict_reverse))])
            g.nodes["user"].data["mask"] = F.one_hot(torch.tensor(0), len(u_dict)).to(torch.bool)
            g.nodes["u_sem"].data["id"] = torch.tensor(u_quant_idx)

            return g

        hist_ids = self.hist_ids[idx]
        hist_mask = self.hist_mask[idx].astype(bool)
        hist_ids = hist_ids[hist_mask]
        hist_ids = list(set(hist_ids))

        hist_ids = np.random.choice(hist_ids, min(self.args.degree, len(hist_ids)), replace=False)

        # remap item quant idx
        i_quant_idx = self.i_quant_idx[hist_ids]  # (N, 3)
        quant_dict = dict()
        for quant_idx in i_quant_idx:
            for idx in quant_idx:
                if idx not in quant_dict:
                    quant_dict[idx] = len(quant_dict)

        quant_dict_reverse = {v: k for k, v in quant_dict.items()}        

        src_s2i = [quant_dict[n] for n in i_quant_idx.reshape(-1)]
        dst_s2i = np.arange(len(hist_ids)).repeat(i_quant_idx.shape[1])


        # remap item id
        i_neighbors = [self.visible_i_code_neighbors[i] for i in i_quant_idx.reshape(-1)]
        i_dict = {hist_id: i for i, hist_id in enumerate(hist_ids)}
        for neighbors in i_neighbors:
            for n in neighbors:
                if n not in i_dict:
                    i_dict[n] = len(i_dict)
        
        i_dict_reverse = {v: k for k, v in i_dict.items()}
        src_i2s = [i_dict[i] for quant_idx in quant_dict for i in self.visible_i_code_neighbors[quant_idx]]
        dst_i2s = np.concatenate([[remap_idx]*len(self.visible_i_code_neighbors[quant_idx]) for quant_idx, remap_idx in quant_dict.items()])

        edges_dict = {
            ("item", "clicked_by", "user"): (
                torch.arange(len(hist_ids)),
                torch.tensor([0]*len(hist_ids))
            ),

            ("u_sem", "denotes", "user"): (
                torch.arange(len(u_quant_idx)),
                torch.tensor([0]*len(u_quant_idx)), 
            ),

            ("user", "shares", "u_sem"): (
                torch.tensor(src_u2s),
                torch.tensor(dst_u2s)
            ),

            ("i_sem", "denotes", "item"): (
                torch.tensor(src_s2i),
                torch.tensor(dst_s2i)
            ),

            ("item", "shares", "i_sem"): (
                torch.tensor(src_i2s), 
                torch.tensor(dst_i2s)
            )
        }

        g = dgl.heterograph(edges_dict)
        g.nodes["user"].data["id"] = torch.tensor([u_dict_reverse[i] for i in range(len(u_dict_reverse))])
        g.nodes["user"].data["mask"] = F.one_hot(torch.tensor(0), len(u_dict)).to(torch.bool)
        g.nodes["item"].data["id"] = torch.tensor([i_dict_reverse[i] for i in range(len(i_dict_reverse))])
        g.nodes["u_sem"].data["id"] = torch.tensor(u_quant_idx)
        g.nodes["i_sem"].data["id"] = torch.tensor([quant_dict_reverse[i] for i in range(len(quant_dict_reverse))])

        return g


    def construct_i_graph(self, iid):
        # remap item id
        i_dict = {iid: 0}
        i_quant_idx = self.i_quant_idx[iid]

        i_neighbors = [self.visible_i_code_neighbors[i] for i in i_quant_idx]
        for neighbors in i_neighbors:
            for n in neighbors:
                if n not in i_dict:
                    i_dict[n] = len(i_dict)
        
        i_dict_reverse = {v: k for k, v in i_dict.items()}

        src_i2s = [i_dict[n] for neighbors in i_neighbors for n in neighbors]
        dst_i2s = np.arange(len(i_quant_idx)).repeat([len(neighbors) for neighbors in i_neighbors])

        if self.args.remove_iu:
            edges_dict = {
                ("i_sem", "denotes", "item"): (
                    torch.arange(len(i_quant_idx)),
                    torch.tensor([0]*len(i_quant_idx)), 
                ),

                ("item", "shares", "i_sem"): (
                    torch.tensor(src_i2s),
                    torch.tensor(dst_i2s)
                ),
            }

            g = dgl.heterograph(edges_dict)
            g.nodes["item"].data["id"] = torch.tensor([i_dict_reverse[i] for i in range(len(i_dict_reverse))])
            g.nodes["item"].data["mask"] = F.one_hot(torch.tensor(0), len(i_dict)).to(torch.bool)
            g.nodes["i_sem"].data["id"] = torch.tensor(i_quant_idx)

            return g

        hist_users = np.array(self.item_to_users[iid])
        hist_users = np.random.choice(hist_users, min(self.args.degree, len(hist_users)), replace=False)

        # For cold-start items, plus dummpy nodes
        if len(hist_users) == 0:
            edges_dict = {
                ("i_sem", "denotes", "item"): (
                    torch.arange(len(i_quant_idx)),
                    torch.tensor([0]*len(i_quant_idx)), 
                ),

                ("item", "shares", "i_sem"): (
                    torch.tensor(src_i2s),
                    torch.tensor(dst_i2s)
                ),

                ("user", "clicked", "item"): (
                    torch.tensor([0]),
                    torch.tensor([0])
                ),

                ("u_sem", "denotes", "user"): (
                    torch.tensor([0]),
                    torch.tensor([0])
                ),

                ("user", "shares", "u_sem"): (
                    torch.tensor([0]), 
                    torch.tensor([0])
                )
            }

            g = dgl.heterograph(edges_dict)
            g.nodes["item"].data["id"] = torch.tensor([i_dict_reverse[i] for i in range(len(i_dict_reverse))])
            g.nodes["item"].data["mask"] = F.one_hot(torch.tensor(0), len(i_dict)).to(torch.bool)
            g.nodes["i_sem"].data["id"] = torch.tensor(i_quant_idx)
            g.nodes["u_sem"].data["id"] = torch.tensor([self.u_code_num])
            g.nodes["user"].data["id"] = torch.tensor([len(self.u_quant_idx)])

            return g


        # remap user quant idx
        u_quant_idx = self.u_quant_idx[hist_users]
        quant_dict = dict()
        for quant_idx in u_quant_idx:
            for idx in quant_idx:
                if idx not in quant_dict:
                    quant_dict[idx] = len(quant_dict)
        
        quant_dict_reverse = {v: k for k, v in quant_dict.items()}

        src_s2u = [quant_dict[n] for n in u_quant_idx.reshape(-1)]
        dst_s2u = np.arange(len(hist_users)).repeat(u_quant_idx.shape[1])


        # remap user id
        u_neighbors = [self.visible_u_code_neighbors[i] for i in u_quant_idx.reshape(-1)]
        u_dict = {uid: i for i, uid in enumerate(hist_users)}
        for neighbors in u_neighbors:
            for n in neighbors:
                if n not in u_dict:
                    u_dict[n] = len(u_dict)
        
        u_dict_reverse = {v: k for k, v in u_dict.items()}
        src_u2s = [u_dict[i] for quant_idx in quant_dict for i in self.visible_u_code_neighbors[quant_idx]]
        dst_u2s = np.concatenate([[remap_idx]*len(self.visible_u_code_neighbors[quant_idx]) for quant_idx, remap_idx in quant_dict.items()])

        edges_dict = {
            ("user", "clicked", "item"): (
                torch.arange(len(hist_users)),
                torch.tensor([0]*len(hist_users))
            ),

            ("i_sem", "denotes", "item"): (
                torch.arange(len(i_quant_idx)),
                torch.tensor([0]*len(i_quant_idx)), 
            ),

            ("item", "shares", "i_sem"): (
                torch.tensor(src_i2s),
                torch.tensor(dst_i2s)
            ),

            ("u_sem", "denotes", "user"): (
                torch.tensor(src_s2u),
                torch.tensor(dst_s2u)
            ),

            ("user", "shares", "u_sem"): (
                torch.tensor(src_u2s), 
                torch.tensor(dst_u2s).to(torch.int64)
            )
        }
        g = dgl.heterograph(edges_dict)
        g.nodes["item"].data["id"] = torch.tensor([i_dict_reverse[i] for i in range(len(i_dict_reverse))])
        g.nodes["item"].data["mask"] = F.one_hot(torch.tensor(0), len(i_dict)).to(torch.bool)
        g.nodes["user"].data["id"] = torch.tensor([u_dict_reverse[i] for i in range(len(u_dict_reverse))])
        g.nodes["i_sem"].data["id"] = torch.tensor(i_quant_idx)
        g.nodes["u_sem"].data["id"] = torch.tensor([quant_dict_reverse[i] for i in range(len(quant_dict_reverse))])

        return g


    def remove_u(self, g, uid):
        if self.args.remove_iu or self.split != "train":
            return g
        u_mapped_id = torch.where(g.nodes["user"].data["id"] == uid)[0]
        if len(u_mapped_id) == 0:
            return g
        etype = ("user", "clicked", "item")
        if g.has_edges_between(u_mapped_id, torch.tensor([0]), etype=etype):
            eid = g.edge_ids(u_mapped_id, torch.tensor([0]), etype=etype)
            return dgl.remove_edges(g, eid, etype=etype)
        else:
            return g


def data_collator_fn(features):
    batch = {
        "user_feats": torch.tensor(np.array([f[0] for f in features])).long(),
        "hist_ids": torch.tensor(np.array([f[1] for f in features])).long(),
        "hist_mask": torch.tensor(np.array([f[2] for f in features])).long(),
        "pos_item_ids": torch.tensor(np.array([f[3] for f in features])).long(),
        "neg_item_ids": torch.tensor(np.array([f[4] for f in features])).long(),
    }

    if len(features[0]) == 6:
        batch["u_graphs"] = dgl.batch([f[5] for f in features])
    if len(features[0]) == 7:
        batch["u_graphs"] = dgl.batch([f[5] for f in features])
        batch["i_graphs"] = dgl.batch([f[6] for f in features])

    return batch
