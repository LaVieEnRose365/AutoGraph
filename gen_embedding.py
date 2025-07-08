import os 
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import torch
import argparse
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM


nums = {
    "BookCrossing": {
        "item": 116848,
        "user": 6853
    },
    "ml-1m": {
        "item": 3533,
        "user": 6038
    },
    "amz-books": {
        "item": 8531,
        "user": 6010
    }
}


def collect_text(tokenizer, args):
    prompts = json.load(open(f"{args.data_dir}/prompt_{args.part}.json"))

    klgs = []
    for i in range(0, nums[args.dataset][args.part], args.group_lens):
        txts = json.load(open(f"{args.data_dir}/klg_{args.part}_{i}_{min(i+args.group_lens, nums[args.dataset][args.part])}.json"))
        klgs.extend(txts)
    print(f"collect_knowledge: {len(klgs)}")
    
    inputs = []
    for prompt, klg in zip(prompts, klgs):
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": klg}
        ]
        cur_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        inputs.append(cur_input)

    return inputs


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        padding_side="left",
        # add_eos_token=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    inputs = collect_text(tokenizer, args)

    avg_embeddings = []

    for text in tqdm(inputs):
        input_ids = tokenizer([text], return_tensors="pt", max_length=4096, truncation=True)["input_ids"]
        input_ids = input_ids.cuda()
        with torch.no_grad():
            outputs = model(input_ids, return_dict=True, output_hidden_states=True)
            avg_embeddings.append(
                outputs.hidden_states[-1].detach().to(torch.float).mean(dim=1).cpu().numpy()
            )

    avg_embeddings = np.concatenate(avg_embeddings, axis=0)
    np.save(f"{args.embed_dir}/{args.dataset}/vicuna_{args.part}.npy", avg_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="vicuna-7b-v1.5")
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--data_dir", type=str, default="data/ml-1m/proc_data")
    parser.add_argument("--embed_dir", type=str, default="outputs_llm")
    parser.add_argument("--part", type=str, default="item")
    parser.add_argument("--group_lens", type=int, default=1000)

    args = parser.parse_args()

    set_seed(42)
    main(args)