import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os.path as osp
import pickle
from collections import defaultdict
from tqdm import tqdm
from utils import OpenAI


def plot_radar_chart(leaf_standard, leaf_aad, figure_save_path, upd_type):
    """
    Plots a radar chart comparing two sets of values across multiple dimensions.

    Parameters:
    - leaf_standard: DataFrame containing the standard dataset values.
    - leaf_aad: DataFrame containing the aad dataset values.
    - figure_save_path: Path where the figure should be saved.
    - upd_type: The type of upd, either "aad", "iasd", or "ivqd".
    """
    # Define the order of columns for plotting
    if upd_type in ["aad", "iasd"]:
        columns_order = [
            'split', 'ocr', 'celebrity_recognition', "object_localization",
            "attribute_recognition", "action_recognition", "attribute_comparison",
            "nature_relation", "physical_relation", "social_relation",
            "identity_reasoning", "function_reasoning",
            "physical_property_reasoning", "structuralized_imagetext_understanding",
            "future_prediction", "image_topic", "image_emotion",
            "image_scene", "image_style"
            ]
    elif upd_type == "ivqd":
        columns_order = [
            'split', 'ocr', 'celebrity_recognition', "object_localization",
            "attribute_recognition", "action_recognition", "attribute_comparison",
            "nature_relation", "physical_relation", "social_relation",
            "function_reasoning", "physical_property_reasoning", "image_scene"
        ]

    # Prepare data
    leaf_standard = leaf_standard[columns_order]
    leaf_aad = leaf_aad[columns_order]
    values_aad = np.append(leaf_aad.iloc[0, 1:].values,
                           leaf_aad.iloc[0, 1])
    values_standard = np.append(leaf_standard.iloc[0, 1:].values,
                                leaf_standard.iloc[0, 1])
    categories = leaf_standard.columns[1:]

    # Initialize radar chart
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    angles = [angle + np.pi/2 if angle + np.pi/2 < 2 * np.pi else angle + np.pi/2 - 2 * np.pi for angle in angles]
    angles[-1] = angles[0]
    angles = angles[::-1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values_aad, color='red', linewidth=2, label='aad')
    ax.fill(angles, values_aad, color='red', alpha=0.25)
    ax.plot(angles, values_standard, color='blue', linewidth=2,
            label='standard')
    ax.fill(angles, values_standard, color='blue', alpha=0.25)

    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.savefig(figure_save_path)
    plt.show()


def eval_result_dual(data_main):
    overall = report_acc(data_main, None)
    l2 = report_acc(data_main, 'l2-category')
    leaf = report_acc(data_main, 'category')

    print(overall)
    print(l2)
    print(leaf)

    return overall, l2, leaf


fout = None


# Utils
def double_log(msg, fout=None):
    print(msg)
    if fout is not None:
        fout.write(str(msg) + '\n')
        fout.flush()


def dump(data, f):

    def dump_pkl(data, pth):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth):
        if isinstance(data, pd.DataFrame):
            data.to_json(pth, orient='records', indent=2)
        else:
            json.dump(data, open(pth, 'w'))

    def dump_jsonl(data, f):
        lines = [json.dumps(x, ensure_ascii=False) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f):
        data.to_excel(f, index=False)

    def dump_csv(data, f):
        data.to_csv(f, index=False)

    def dump_tsv(data, f):
        data.to_csv(f, sep='\t', index=False)

    handlers = dict(pkl=dump_pkl,
                    json=dump_json,
                    jsonl=dump_jsonl,
                    xlsx=dump_xlsx,
                    csv=dump_csv,
                    tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f)


def load(f):
    """
        Loads data from various file formats.

        Parameters:
        - file_path: Path to the file to be loaded.

        Returns:
        - Loaded data.
    """
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        df = pd.read_excel(f)
        df = df.dropna(subset=['prediction'])

        return df

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl,
                    json=load_json,
                    jsonl=load_jsonl,
                    xlsx=load_xlsx,
                    csv=load_csv,
                    tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f)


# Accuracy Report
def report_acc(df, groupd='category'):
    assert 'split' in df
    assert groupd in [None, 'category', 'l2-category']

    res = defaultdict(list)
    res['split'] = ['test']
    if groupd is None:
        res['overall'] = [
            np.mean(df['hit']),
        ]
        return pd.DataFrame(res)

    elif groupd in df:
        abilities = list(set(df[groupd]))
        abilities.sort()
        for ab in abilities:
            sub_df = df[df[groupd] == ab]
            res[ab] = [
                np.mean(sub_df['hit']),
            ]
        return pd.DataFrame(res)