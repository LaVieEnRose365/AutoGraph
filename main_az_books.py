import os
import logging
from utils_az_books import parse_args, weight_init
from dataset_az_books import MatchDataset, data_collator_fn
from transformers import set_seed
from models_az_books import MIND, DSSM, GRU4Rec, SASRec, SDM
from trainer import Trainer
import dgl


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    running_log = os.path.join(args.output_dir, "result.log")

    logging.basicConfig(
        format="%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(filename=running_log, mode="w"))


    set_seed(args.seed)
    logger.info(args)

    match_dataset = MatchDataset(args)
    args.num_users = match_dataset.num_users
    args.num_items = match_dataset.num_items
    args.num_user_fields = match_dataset.num_user_fields
    args.num_item_fields = match_dataset.num_item_fields
    args.num_user_feats = match_dataset.num_user_feats
    args.num_item_feats = match_dataset.num_item_feats
    args.item_feats_table = match_dataset.item_feats_table
    if args.plus_llm_embed:
        args.llm_embeds = match_dataset.llm_embeds
        args.llm_embed_dim = match_dataset.llm_embed_dim

    if args.plus_gnn_embed:
        args.u_code_num = match_dataset.u_code_num
        args.i_code_num = match_dataset.i_code_num

    datasets = {split: match_dataset.get_splited_dataset(split) for split in ["train", "valid", "test"]}

    if args.plus_gnn_embed:
        args.i_graphs = dgl.batch(datasets["train"].i_graphs).to("cuda")
        datasets["valid"].i_graphs = datasets["train"].i_graphs
        datasets["test"].i_graphs = datasets["train"].i_graphs

    if args.model_name.lower() == "dssm":
        model = DSSM(args)
    elif args.model_name.lower() == "gru4rec":
        model = GRU4Rec(args)
    elif args.model_name.lower() == "mind":
        model = MIND(args)
    elif args.model_name.lower() == "sasrec":
        model = SASRec(args)
    elif args.model_name.lower() == "sdm":
        model = SDM(args)
    else:
        raise NotImplementedError

    model.apply(weight_init)


    collator = data_collator_fn

    trainer = Trainer(
        model=model, 
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        data_collator_fn=collator,
    )

    trainer.train()
    trainer.test(datasets["test"])



if __name__ == "__main__":
    args = parse_args()
    main(args)

