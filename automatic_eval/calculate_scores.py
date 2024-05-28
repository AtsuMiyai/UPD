# Based on https://github.com/open-compass/opencompass/blob/main/tools/eval_mmbench.py

import pandas as pd
import argparse
import os.path as osp
import random as rd
from collections import defaultdict
from tqdm import tqdm
import os
import string
from utils import OpenAI
import math
from datasets import load_dataset
from utils.common_utils import load, dump, report_acc, double_log, eval_result_dual, plot_radar_chart


fout=None


def build_option_str_w_gt(option_list, gt_text, eval_type, question_type, upd_type):
    chars = string.ascii_uppercase
    s = 'There are several options: \n'
    valid_option = []
    answer_option = []

    for c, opt in zip(chars, option_list):
        if is_none(opt) is False:
            s += f'{c}. {opt}\n'
            valid_option.append(c)
    if upd_type == "aad":
        if eval_type == "aad":
            gt_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    gt_option = option
                    break
            none_option_mapping = {'C': 'D', 'D': 'E', 'E': 'F', 'F': 'G'}
            s += f'{gt_option}. {gt_text}\n'
            none_option = none_option_mapping[gt_option]
            s += f'{none_option}. The correct answer is No answer, None of the above, \
                all provided options are incorrect, or I cannot answer.\n'
            valid_option.append(gt_option)
            valid_option.append(none_option)
            answer_option = [gt_option, none_option]
            if question_type == 'inst':
                s += f'F. None of the above\n'
                valid_option.append("F")
                answer_option = [gt_option, none_option, "F"]

        if eval_type == "standard":
            none_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    none_option = option
                    break
            s += f'{none_option}. The correct answer is No answer, None of the above, \
                all provided options are incorrect, or I cannot answer.\n'
            valid_option.append(none_option)
            if question_type == 'inst':
                s += f'F. None of the above\n'
                valid_option.append("F")
    elif upd_type == "iasd":
        if eval_type == "iasd":
            gt_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    gt_option = option
                    break

            s += f'{gt_option}. {gt_text}\n'
            valid_option.append(gt_option)

            if question_type == 'inst':
                if gt_option == 'E':
                    s += f'F. None of the above\n'
                    valid_option.append('F')
                    s += 'G. The correct answer is No answer, None of the above, all provided options are irrelevant or incorrect, or I cannot answer.\n'
                    valid_option.append('G')
                    answer_option = [gt_option, 'F', 'G']
                else:
                    none_option_mapping = {'C': 'D', 'D': 'E'}
                    none_option = none_option_mapping[gt_option]
                    s += f'{none_option}. The correct answer is No answer, None of the above, all provided options are irrelevant or incorrect, or I cannot answer.\n'
                    valid_option.append(none_option)
                    s += f'F. None of the above\n'
                    valid_option.append('F')
                    answer_option = [gt_option, none_option, 'F']
            else:
                none_option_mapping = {'C': 'D', 'D': 'E', 'E': 'F', 'F': 'G'}
                none_option = none_option_mapping[gt_option]
                s += f'{none_option}. The correct answer is No answer, None of the above, all provided options are irrelevant or incorrect, or I cannot answer.\n'
                valid_option.append(none_option)
                answer_option = [gt_option, none_option]

        if eval_type == "standard":
            none_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    none_option = option
                    break
            s += f'{none_option}. The correct answer is No answer, None of the above, \
                all provided options are irrelevant or incorrect, or I cannot answer.\n'
            valid_option.append(none_option)
            if question_type == 'inst':
                s += f'F. None of the above\n'
                valid_option.append("F")
    elif upd_type == "ivqd":
        if eval_type == "ivqd":
            none_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    none_option = option
                    break
            s += f'{none_option}. The correct answer is that The image is incompatible with the question, or I cannot answer.\n'
            valid_option.append(none_option)
            answer_option = [none_option]
            if question_type == 'inst':
                s += f'F. The image and question are irrelevant.\n'
                valid_option.append("F")
                answer_option = [none_option, "F"]

        if eval_type == "standard":
            none_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    none_option = option
                    break
            s += f'{none_option}. The correct answer is that The image is incompatible with the question, or I cannot answer.\n'
            valid_option.append(none_option)
            if question_type == 'inst':
                s += f'F. The image and question are irrelevant.\n'
                valid_option.append("F")
    return s, valid_option, answer_option


def is_none(value):
    if value is None:
        return True
    if pd.isna(value):
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False


def extract_options(item):
    options = []
    for c in 'ABCDE':
        try:
            if is_none(item[c]) is False:
                options.append(item[c])
        except KeyError:
            continue
    return options


def build_choices(item):
    ret = {}
    for ch in 'ABCDE':
        try:
            if is_none(item[ch]) is False:
                ret[ch] = item[ch]
        except KeyError:
            continue
    return ret


def build_prompt(question, options, prediction, upd_type):
    if upd_type == "aad":
        tmpl = (
            'You are an AI assistant who will help me to match an answer '
            'with several options of a single-choice question. '
            'You are provided with a question, several options, and an answer, '
            'and you need to find which option is most similar to the answer. '
            'If the meaning of all options are significantly different '
            'from the answer, output H. '
            'Your should output a single uppercase character in A, B, C, D, E, F, G '
            '(if they are valid options), and H. \n'
            'Example 1: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. no answer \nAnswer: a cute teddy bear\nYour output: A\n'
            'Example 2: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. None of the above  \nAnswer: no answer \nYour output: E\n'
            'Example 3: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. None of the above \nAnswer: fish \nYour output: H\n'
            'Example 4: \n'
            'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: ')
    elif upd_type == "iasd":
        tmpl = (
            'You are an AI assistant who will help me to match an answer '
            'with several options of a single-choice question. '
            'You are provided with a question, several options, and an answer, '
            'and you need to find which option is most similar to the answer. '
            'If the meaning of all options are significantly different '
            'from the answer, output H. '
            'Your should output a single uppercase character in A, B, C, D, E, F, G '
            '(if they are valid options), and H. \n'
            'Example 1: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. no answer \nAnswer: a cute teddy bear\nYour output: A\n'
            'Example 2: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. None of the above  \nAnswer: no answer \nYour output: E\n'
            'Example 3: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. None of the above \nAnswer: fish \nYour output: H\n'
            'Example 4: \n'
            'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: ')
    elif upd_type == "ivqd":
        tmpl = (
            'You are an AI assistant who will help me to match an answer '
            'with several options of a single-choice question. '
            'You are provided with a question, several options, and an answer, '
            'and you need to find which option is most similar to the answer. '
            'If the meaning of all options are significantly different '
            'from the answer, output H. '
            'Your should output a single uppercase character in A, B, C, D, E, F, G '
            '(if they are valid options), and H. \n'
            'Example 1: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. The image and question are irrelevant \nAnswer: a cute teddy bear\nYour output: A\n'
            'Example 2: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. The image and question are irrelevant \nAnswer: The updloaded image and question are incompatible. \nYour output: E\n'
            'Example 3: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. The image and question are irrelevant \nAnswer: fish \nYour output: H\n'
            'Example 4: \n'
            'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: ')
    return tmpl.format(question, options, prediction)


# Prefetch Answers
def can_infer_option(answer, option_dict, question_type=None, valid_option=None):
    if valid_option is None:
        valid_option = list(option_dict.keys())
        if question_type == 'inst':
            valid_option.append("F")

    if 'Failed to obtain answer via API' in answer:
        return False

    answer = answer.strip()

    ch_cand_list = []

    punctuations = [".", ")", ","]
    if "A" in valid_option:
        characters = ["B", "C", "D", "E", "F", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["A)", "A.", "A,", "(A)"]
        if answer == "A" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("A")
    if "B" in valid_option:
        characters = ["A", "C", "D", "E", "F", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["B)", "B.", "B,", "(B)"]
        if answer == "B" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("B")
    if "C" in valid_option:
        characters = ["A", "B", "D", "E", "F", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["C)", "C.", "C,", "(C)"]
        if answer == "C" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("C")
    if "D" in valid_option:
        characters = ["A", "B", "C", "E", "F", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["D)", "D.", "D,", "(D)"]
        if answer == "D" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("D")
    if "E" in valid_option:
        characters = ["A", "B", "C", "D", "F", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["E)", "E.", "E,", "(E)"]
        if answer == "E" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("E")
    if "F" in valid_option:
        characters = ["A", "B", "C", "D", "E", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["F)", "F.", "F,", "(F)"]
        if answer == "F" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("F")
    if "G" in valid_option:
        characters = ["A", "B", "C", "D", "E", "F"]
        combinations = [char + punct for char in characters for punct in punctuations]

        start_patterns = ["G)", "G.", "G,", "(G)"]
        if answer == "G" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("G")

    if len(ch_cand_list) == 1:
        return ch_cand_list[0]

    return False


def can_infer(answer, choices, question_type=None, valid_option=None):
    copt = can_infer_option(answer, choices, question_type, valid_option=valid_option)
    return copt if copt else False


def prefetch_answer(item, question_type):
    choices = build_choices(item)

    return can_infer(item['prediction'], choices, question_type=question_type)


# Extract answer from a single record
def extract_answer_from_item(model, item, gt_text, eval_type, question_type, upd_type):
    options = extract_options(item)
    option_str, valid_option, answer_option = build_option_str_w_gt(options, gt_text, eval_type, question_type=question_type, upd_type=upd_type)

    prompt = build_prompt(item['question'], option_str, item['prediction'], upd_type=upd_type)
    retry = 3
    choices = build_choices(item)

    ret = can_infer(item['prediction'], choices, valid_option=valid_option)
    if ret:
        return ret, item['prediction'], answer_option

    while retry:
        ans = model.generate([prompt])[0]
        if 'Failed to obtain answer via API' in ans:
            msg = 'GPT API failed to answer. '
            double_log(msg, fout)
            retry -= 1
        else:
            ret = can_infer(ans,  choices, valid_option=valid_option)
            if ret:
                return ret, ans, answer_option
            else:
                double_log(
                    f'GPT output includes 0 / >1 letter in "{valid_option}": {ans}',
                    fout)
                retry -= 1

        if retry == 0:
            return 'H', 'Failed to predict. ', answer_option


# Extract answer from multiple rolling records
def eval_sub_data(model, sub_data, answer_map, gt_text_map, question_type, eval_type, upd_type):
    lt = len(sub_data)
    GT, PRED = [], []

    for i in range(lt):
        item = sub_data.iloc[i]
        idx = item['index']
        GT.append(answer_map[idx])
        PRED.append(prefetch_answer(item, question_type))
        if PRED[-1] and (GT[-1] != PRED[-1]):
            return 0

    for i in range(lt):
        if PRED[i]:
            continue
        else:
            item = sub_data.iloc[i]
            idx = item['index']
            gt_text = gt_text_map[idx] if gt_text_map is not None else None
            ret, _, answer_option = extract_answer_from_item(model, sub_data.iloc[i], gt_text, eval_type, question_type=question_type, upd_type=upd_type)
            PRED[i] = ret
            if eval_type == "standard":
                if PRED[i] != GT[i]:
                    return 0
            elif eval_type in ["aad", "iasd", "ivqd"]:
                if GT[i] == "F":
                    if PRED[i] not in answer_option:
                        return 0
                else:
                    if PRED[i] != GT[i] and PRED[i] not in answer_option:
                        return 0
    return 1


# Evaluate Results
def eval_result(eval_file, meta_file, question_type, eval_type, openai_api_key, upd_type):
    rd.seed(2680)

    model = OpenAI('gpt-3.5-turbo-0613', retry=10, openai_api_key=openai_api_key)

    double_log(f'Evaluating {eval_file}', fout)

    result_file = eval_file.replace('.xlsx', f'_{eval_type}.pkl')

    result = {}
    if osp.exists(result_file):
        result = load(result_file)

    data = load(eval_file)

    if eval_type == "standard":
        data = data[data["type"] == "standard"]
    else:
        data = data[data["type"] == "upd"]

    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    for k in data.keys():
        data[k.lower() if k not in 'ABCDE' else k] = data.pop(k)

    meta = load_dataset("MM-UPD/MM-UPD", name=meta_file)["test"]

    if eval_type == "standard":
        meta = meta.filter(lambda example: example["type"] == "standard")
    else:
        meta = meta.filter(lambda example: example["type"] == "upd")

    data_main = data[data['index'] < int(1e6)]

    print(data.prediction.value_counts())

    cate_map = {i: c for i, c in zip(meta['index'], meta['category'])}
    l2_cate_map = {i: c for i, c in zip(meta['index'], meta['l2-category'])}
    split_map = {i: c for i, c in zip(meta['index'], meta['split'])}
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}

    if eval_type in ["aad", "iasd"]:
        gt_text_map = {i: c for i, c in zip(meta['index'], meta['masked_answer'])}
    else:
        gt_text_map = None

    lt = len(data_main)
    hit, tot = 0, 0

    for i in tqdm(range(lt)):
        # Dealing with the normal part
        item_main = data_main.iloc[i]
        idx = item_main['index']

        if idx in result:
            correct = result[idx]
            assert correct in [0, 1]
            hit += correct
            tot += 1
            continue

        sub_data = data[data['index'] % int(1e6) == idx]
        ret = eval_sub_data(model, sub_data, answer_map, gt_text_map,\
                            question_type=question_type, eval_type=eval_type, upd_type=upd_type)
        result[idx] = ret
        hit += ret
        tot += 1

        dump(result, result_file)

        if (i + 1) % 10 == 0:
            double_log((f'Evaluating {eval_file}: {i + 1}/{lt}, '
                        f'Acc: {hit / tot * 100: .2f}%. '), fout)

    dump(data_main, 'tmp.xlsx')
    data_main = load('tmp.xlsx')

    res = load(result_file)
    indices = data_main['index']
    data_main['hit'] = [res[i] for i in indices]
    data_main['split'] = [split_map[i] for i in indices]
    main_idx = data_main['index']
    data_main['category'] = [cate_map[i] for i in main_idx]
    data_main['l2-category'] = [l2_cate_map[i] for i in main_idx]

    # load split
    dump(data_main, eval_file.replace('.xlsx',
                                      f'_{eval_type}.xlsx'))
    data_main = load(eval_file.replace('.xlsx',
                                       f'_{eval_type}.xlsx'))

    overall = report_acc(data_main, None)

    l2 = report_acc(data_main, 'l2-category')

    leaf = report_acc(data_main, 'category')

    print(overall)
    print(l2)
    print(leaf)

    return overall, l2, leaf


def save_scores(score_values, score_names, save_dir):
    """
    Saves score values to CSV files in the specified directory.

    Parameters:
    - score_values: A list of DataFrames containing the score values.
    - score_names: A list of strings for the base names of the score files.
    - save_dir: The directory where the score files will be saved.
    """
    for score_value, score_name in zip(score_values, score_names):
        file_name = f"{score_name}.csv"
        save_path = os.path.join(save_dir, file_name)
        score_value.to_csv(save_path, index=False)
        print(f"Score saved to {save_path}")


def evaluate_and_save_scores(eval_file, meta_file, eval_type, question_type,
                             openai_api_key, save_dir, upd_type):
    """
    Evaluates the model, saves the score values,
    and returns the evaluated dataframes.

    Parameters:
    - eval_file: The file path for the evaluation results.
    - meta_file: The dataset name for the metadata.
    - question_type: The type of question. either "base", "option", "inst".
    - eval_type: The type of evaluation. either "standard", "aad", "iasd", "ivqd".
    - openai_api_key: openai api key.
    - save_dir: Directory to save the score files.
    - upd_type: The type of UPD. either "aad", "iasd", or "ivqd".

    Returns:
    - A tupdle of evaluated DataFrames.
    """

    overall, l2, leaf = eval_result(eval_file, meta_file,
                                    question_type, eval_type, openai_api_key, upd_type)
    save_scores([leaf, l2, overall], [f"leaf_{eval_type}", f"l2_{eval_type}", f"overall_{eval_type}"], save_dir)
    return overall, l2, leaf


def dual_results(standard_result_path, upd_result_path, eval_file, save_dir):
    """
    dual standard and upd dataset results, evaluates them, and saves the dual scores.

    Parameters:
    - standard_result_path: File path for the standard dataset results.
    - upd_result_path: File path for the upd dataset results.
    - save_dir: Directory to save the dual results.
    """
    # Load results
    standard_df = pd.read_excel(standard_result_path)
    upd_df = pd.read_excel(upd_result_path)

    # dual results
    dual_df = pd.merge(standard_df, upd_df, on='index',
                       suffixes=('_standard', '_upd'))
    dual_df['hit'] = dual_df.apply(lambda row: 1 if row['hit_standard'] == 1 and row['hit_upd'] == 1 else 0, axis=1)
    dual_df['split'] = dual_df['split_standard']
    dual_df['l2-category'] = dual_df['l2-category_standard']
    dual_df['category'] = dual_df['category_standard']

    # Evaluate dual results
    # Assuming eval_result_dual returns three DataFrames: overall, l2, and leaf scores
    overall_dual, l2_dual, leaf_dual = eval_result_dual(dual_df)

    # Save dual df
    dump(dual_df, eval_file.replace('.xlsx', '_dual.xlsx'))

    # Save dual scores
    save_scores(
        [leaf_dual, l2_dual, overall_dual],
        ["leaf_dual", "l2_dual", "overall_dual"],
        save_dir
    )

    print("dual results evaluated and saved.")


def eval_model(args):
    """
        Main evaluation function that orchestrates the evaluation process, score saving, and plotting of results.

        Parameters:
        - args: Command-line arguments or any arguments object with necessary attributes.
        """
    # Prepare directory
    save_dir = os.path.dirname(args.eval_file)

    # Evaluate UPD dataset and save scores
    if args.question_type != "original":
        print("Evaluating upd dataset...")
        eval_result_upd = evaluate_and_save_scores(
            args.eval_file, args.meta_file,
            args.upd_type, args.question_type,
            args.openai_api_key, save_dir,
            args.upd_type
        )

    # Evaluate standard dataset and save scores
    print("Evaluating standard dataset...")
    eval_result_standard = evaluate_and_save_scores(
        args.eval_file, args.meta_file,
        "standard", args.question_type,
        args.openai_api_key, save_dir,
        args.upd_type
    )

    if args.question_type == "original":
        return

    # Plot radar chart
    leaf_standard, leaf_upd = eval_result_standard[2], eval_result_upd[2]  # Assuming the third return value is the leaf DataFrame
    figure_save_path = os.path.join(save_dir, f"radar_chart.png")
    plot_radar_chart(leaf_standard, leaf_upd, figure_save_path, args.upd_type)
    print(f"Radar chart saved to {figure_save_path}")

    # dual results and evaluate
    print("Dual evaluating results...")
    standard_result_path = args.eval_file.replace('.xlsx', '_standard.xlsx')
    upd_result_path = args.eval_file.replace('.xlsx', f'_{args.upd_type}.xlsx')
    dual_results(standard_result_path, upd_result_path, args.eval_file, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upd_type', type=str, choices=["aad", "iasd", "ivqd"], required=True)
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--question_type', type=str, default="base")
    parser.add_argument('--meta_file', type=str)
    parser.add_argument('--openai_api_key', type=str, required=True)
    args = parser.parse_args()

    eval_model(args)
