import os
import pandas as pd
import argparse
import json
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from tqdm import trange
import torch


def BX_user_data(tokenizer, args):
    prompt = (
        "Given a user who is {age} and in {location}, this user's book reading history over time is listed as below: \n{user_hist}.\n"
        "Analyze the user's preferences (consider factors like genre, author, characters, plot, topic/theme, writing style, award/critical acclaim, etc.)."
        "Provide clear explanations based on details from the user's reading history and other pertinent factors."
    )

    df = pd.read_parquet(os.path.join(args.data_dir, "valid.parquet.gz"))
    meta = json.load(open(os.path.join(args.data_dir, "match-meta.json")))
    feature_dict = meta["feature_dict"]
    book_feats_table = meta["book_feats_table"]
    title_dict = feature_dict["Book title"]
    reverse_title_dict = {int(v): k for k, v in title_dict.items()}
    age_dict = feature_dict["Age"]
    reverse_age_dict = {int(v): k for k, v in age_dict.items()}
    location_dict = feature_dict["Location"]
    reverse_location_dict = {int(v): k for k, v in location_dict.items()}

    inputs = []
    for _, row in df.iterrows():
        age = reverse_age_dict[row["Age"]]
        location = reverse_location_dict[row["Location"]]
        user_hist = ", ".join([f"{idx}. " + reverse_title_dict[book_feats_table[i][1]] for idx, i in enumerate(row["user history ID"][-30:])])
        conversation = [
            {"role": "user", "content": prompt.format(age=age, location=location, user_hist=user_hist)},
        ]

        cur_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs.append(cur_input)

    split = args.split.split(":")
    start = int(split[0])
    end = int(split[1]) if split[1] != "" else len(inputs)

    return inputs, start, end


def BX_item_data(tokenizer, args):
    prompt = (
        "Introduce book {title} and describe its attributes precisely "
        "(including but not limited to genre, author, characters, plot, topic/theme, writing style, award/critical acclaim, etc.)."
    )

    meta = json.load(open(os.path.join(args.data_dir, "match-meta.json")))
    title_dict = meta["feature_dict"]["Book title"]
    reverse_title_dict = {int(v): k for k, v in title_dict.items()}

    inputs = []
    for i in range(len(reverse_title_dict)):
        conversation = [
            {"role": "user", "content": prompt.format(title=reverse_title_dict[i])},
        ]

        cur_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs.append(cur_input)
    
    split = args.split.split(":")
    start = int(split[0])
    end = int(split[1]) if split[1] != "" else len(inputs)

    return inputs, start, end


def ml25m_user_data(tokenizer, args):
    prompt = (
        "A user's movie viewing history over time is listed below: \n{user_hist}.\n"
        "Analyze the user's preferences on movies (consider factors like genre, director/actors, time "
        "period/country, character, plot/theme, mood/tone, critical acclaim/award, production quality, "
        "and soundtrack). Provide clear explanations based on relevant details from the user\'s movie "
        "viewing history and other pertinent factors."
    )

    df = pd.read_parquet(os.path.join(args.data_dir, "valid.parquet.gz"))
    meta = json.load(open(os.path.join(args.data_dir, "match-meta.json")))
    feature_dict = meta["feature_dict"]
    title_dict = feature_dict["Movie title"]
    reverse_title_dict = {int(v): k for k, v in title_dict.items()}
    movie_feats_table = meta["movie_feats_table"]

    inputs = []
    for _, row in df.iterrows():
        user_hist = "; ".join([f"{idx}. " + reverse_title_dict[movie_feats_table[i][1]] for idx, i in enumerate(row["user history ID"][-30:])])
        conversation = [
            {"role": "user", "content": prompt.format(user_hist=user_hist)},
        ]

        cur_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs.append(cur_input)

    split = args.split.split(":")
    start = int(split[0])
    end = int(split[1]) if split[1] != "" else len(inputs)

    return inputs, start, end


def ml25m_item_data(tokenizer, args):
    prompt = (
        "Introduce movie {title} and describe its attributes (including but not limited to genre, "
        "director/cast, country, character, plot/theme, mood/tone, critical "
        "acclaim/award, production quality, and soundtrack)."
    )

    meta = json.load(open(os.path.join(args.data_dir, "match-meta.json"), encoding="utf-8"))
    title_dict = meta["feature_dict"]["Movie title"]
    reverse_title_dict = {int(v): k for k, v in title_dict.items()}

    movie_feats_table = meta["movie_feats_table"]

    inputs = []
    for i in range(len(movie_feats_table)):
        title_id = movie_feats_table[i][1]
        conversation = [
            {"role": "user", "content": prompt.format(title=reverse_title_dict[title_id])},
        ]

        cur_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs.append(cur_input)

    split = args.split.split(":")
    start = int(split[0])
    end = int(split[1]) if split[1] != "" else len(inputs)

    return inputs, start, end


def az_user_data(tokenizer, args):
    prompt = (
        "A user's movie viewing history over time is listed below: \n{user_hist}.\n"
        "Analyze the user's preferences on movies (consider factors like genre, director/actors, time "
        "period/country, character, plot/theme, mood/tone, critical acclaim/award, production quality, "
        "and soundtrack). Provide clear explanations based on relevant details from the user\'s movie "
        "viewing history and other pertinent factors."
    )

    df = pd.read_parquet(os.path.join(args.data_dir, "valid.parquet.gz"))
    title_dict = json.load(open("/NAS2020/Workspaces/DMGroup/rongshan/GNN_LLM/data/az-books/proc_data/datamaps.json"))["itemid2title"]
    title_dict = {int(k): v for k, v in title_dict.items()}

    inputs = []
    for _, row in df.iterrows():
        user_hist = "; ".join([f"{idx}. " + title_dict[i] for idx, i in enumerate(row["user history ID"][-30:])])
        conversation = [
            {"role": "user", "content": prompt.format(user_hist=user_hist)},
        ]

        cur_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs.append(cur_input)

    split = args.split.split(":")
    start = int(split[0])
    end = int(split[1]) if split[1] != "" else len(inputs)

    return inputs, start, end



def az_item_data(tokenizer, args):
    prompt = (
        "Introduce movie {title} and describe its attributes (including but not limited to genre, "
        "director/cast, country, character, plot/theme, mood/tone, critical "
        "acclaim/award, production quality, and soundtrack)."
    )

    title_dict = json.load(open("/NAS2020/Workspaces/DMGroup/rongshan/GNN_LLM/data/az-books/proc_data/datamaps.json"))["itemid2title"]
    title_dict = {int(k): v for k, v in title_dict.items()}

    inputs = []
    for i in range(1, len(title_dict)+1):
        conversation = [
            {"role": "user", "content": prompt.format(title=title_dict[i])},
        ]

        cur_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs.append(cur_input)

    split = args.split.split(":")
    start = int(split[0])
    end = int(split[1]) if split[1] != "" else len(inputs)

    return inputs, start, end






def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        padding_side="left",   
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    llm  = LLM(args.model_path, tensor_parallel_size=args.num_gpus, gpu_memory_utilization=args.gpu_ratio)

    if args.dataset == "BookCrossing":
        if args.part == "user":
            texts, start, end = BX_user_data(tokenizer, args)
        else:
            texts, start, end = BX_item_data(tokenizer, args)

    elif args.dataset == "ml-25m":
        if args.part == "user":
            texts, start, end = ml25m_user_data(tokenizer, args)
        else:
            texts, start, end = ml25m_item_data(tokenizer, args)
    elif args.dataset == "az-books":
        if args.part == "user":
            texts, start, end = az_user_data(tokenizer, args)
        else:
            texts, start, end = az_item_data(tokenizer, args)

    texts = texts[start:end]
    print(texts[0])
    exit()
    print(f"Data prepared, range {start}:{end}.")

    # terminators = [
    #     tokenizer.eos_token_id,
    #     tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]

    # sampling_params = SamplingParams(temperature=0.9, top_p=0.6, max_tokens=args.max_tokens, stop_token_ids=terminators)
    sampling_params = SamplingParams(temperature=0.9, top_p=0.6, max_tokens=args.max_tokens)

    generated_texts = []
    saved_groups = 0
    group_lens = 5120 * 2

    for i in trange(0, len(texts), args.batch_size):
        outputs = llm.generate(texts[i:i+args.batch_size], sampling_params, use_tqdm=False)

        for output in outputs:
            generated_texts.append(output.outputs[0].text)

        if len(generated_texts) == group_lens:
            cur_start = start + saved_groups * group_lens
            cur_end = cur_start + group_lens
            output_path = f"{args.embed_dir}/{args.dataset}/txt_{args.part}_{cur_start}_{cur_end}.json"
            json.dump(generated_texts, open(output_path, "w"), indent=2)
            generated_texts = []
            saved_groups += 1
    
    if len(generated_texts) != 0:
        output_path = f"{args.embed_dir}/{args.dataset}/txt_{args.part}_{cur_end}_{end}.json"
        json.dump(generated_texts, open(output_path, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--embed_dir", type=str, default="../outputs_llm")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--part", type=str, default="item")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--split", type=str, default="0:")
    parser.add_argument("--gpu_ratio", type=float, default=0.9)
    args = parser.parse_args()

    args.dataset = args.data_dir.split("/")[-2]
    os.makedirs(f"{args.embed_dir}/{args.dataset}", exist_ok=True)

    set_seed(42)
    main(args)