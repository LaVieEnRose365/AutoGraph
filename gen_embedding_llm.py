import os
import json
import argparse
from transformers import set_seed
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


def prepare_item(args):
    prompt = (
        "Here is a movie. Its title is {title}. The movie's genre is {genre}."
    )

    meta = json.load(open(os.path.join(args.data_dir, "match-meta.json")))
    title_dict = meta["feature_dict"]["Movie title"]
    reverse_title_dict = {int(v): k for k, v in title_dict.items()}
    genre_dict = meta["feature_dict"]["Movie genre"]
    reverse_genre_dict = {int(v): k for k, v in genre_dict.items()}

    movie_feats_table = meta["movie_feats_table"]

    inputs = []
    for i in range(len(reverse_title_dict)):
        title = reverse_title_dict[movie_feats_table[i][1]]
        genre = reverse_genre_dict[movie_feats_table[i][2]]
        inputs.append(
            prompt.format(title=title, genre=genre)
        )

    return inputs


def prepare_user(args):
    prompt = (
        "The user is a {gender}. {ptr} job is {job}. {ptr} age is {age}. The user watched the following movies in order in the past:\n"
        "{hist}."
    )

    df = pd.read_parquet(os.path.join(args.data_dir, "valid.parquet.gz"))

    meta = json.load(open(os.path.join(args.data_dir, "match-meta.json")))
    feature_dict = meta["feature_dict"]

    user_num = len(feature_dict["User ID"])

    title_dict = feature_dict["Movie title"]
    reverse_title_dict = {int(v): k for k, v in title_dict.items()}
    gender_dict = feature_dict["Gender"]
    reverse_gender_dict = {int(v): k for k, v in gender_dict.items()}
    age_dict = feature_dict["Age"]
    reverse_age_dict = {int(v): k for k, v in age_dict.items()}
    job_dict = feature_dict["Job"]
    reverse_job_dict = {int(v): k for k, v in job_dict.items()}

    txt_dict = {}
    for _, row in df.iterrows():
        gender = reverse_gender_dict[row["Gender"]]
        age = reverse_age_dict[row["Age"]]
        job = reverse_job_dict[row["Job"]]
        ptr = "Her" if gender == "female" else "His"
        hist = ", ".join([f"{idx}. " + reverse_title_dict[i] for idx, i in enumerate(row["user history ID"][-30:])])
        txt_dict[row["User ID"]] = prompt.format(
            gender=gender,
            job=job, ptr=ptr,
            age=age, hist=hist
        )

    inputs = [txt_dict[i] if i in txt_dict else "" for i in range(user_num)]

    return inputs


def main(args):
    if args.part == "item":
        data = prepare_item(args)
    else:
        data = prepare_user(args)

    model = SentenceTransformer(args.model_path, device="cuda")
    embeddings = model.encode(data, batch_size=args.batch_size, show_progress_bar=True)
    np.save(os.path.join(args.output_path, args.dataset, f"{args.model_path.split('/')[-1]}_{args.part}.npy"), embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="../outputs_llm")
    parser.add_argument("--part", type=str, default="item")
    args = parser.parse_args()
    
    args.dataset = args.data_dir.split("/")[-2]
    set_seed(42)
    main(args)
