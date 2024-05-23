import os
import json
import argparse
import pandas as pd
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", type=str, default="mmaad_base")
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    questions = load_dataset("MM-UPD/MM-UPD", name=args.data_name)["test"]
    df = questions.to_pandas()

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image',\
                                  'comment', 'l2-category'])

    e_index = cur_df.columns.get_loc('E') + 1
    cur_df.insert(e_index, 'prediction', None)

    for pred in open(os.path.join(args.result_dir, f"{args.experiment}.jsonl")):
        pred = json.loads(pred)
        cur_df.loc[(df['index'] == pred['question_id']) & (df['type'] == pred['eval_type']), 'prediction'] = pred['text']

    cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"),\
                    index=False, engine='openpyxl')
