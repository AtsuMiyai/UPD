# Based on https://huggingface.co/THUDM/cogvlm-chat-hf

import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
import math
from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image
from io import BytesIO
import base64
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


all_options = ['A', 'B', 'C', 'D', 'E']


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


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


def get_options(row, options):
    parsed_options = []
    for option in options:
        try:
            option_value = row[option]
        except KeyError:
            break
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def read_jsonl(file_path):
    with open(file_path, 'r') as json_file:
        return [json.loads(line) for line in json_file]


def eval_model(args):
    questions = load_dataset("MM-UPD/MM-UPD", name=args.data_name)["test"]                                              
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TORCH_TYPE = torch.half  # torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
    ).to(DEVICE).eval()


    for row in tqdm(questions, total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question

            if args.single_pred_prompt:
                if args.prompt_id == 0:
                    qs = qs + '\n'
                if args.prompt_id == 1:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                elif args.prompt_id == 2:
                    qs = qs + '\n' + "If all the options are incorrect, answer \"F. None of the above\"."
                elif args.prompt_id == 3:
                    qs = qs + '\n' + "If the given image is irrelevant to the question, answer \"F. The image and question are irrelevant.\"."

            history = []
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=qs,
                history=history,
                images=[image],
                template_version='vqa'
            )

            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to('cuda'),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to('cuda'),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to('cuda'),
                'images': [[input_by_model['images'][0].to('cuda').to(torch.float16)]] if image is not None else None,
            }
            gen_kwargs = {
                "max_new_tokens": 2048,
                "pad_token_id": 128002,
                "do_sample": False
            }

            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                outputs = tokenizer.decode(outputs[0])
            outputs = outputs.rsplit("<|end_of_text|>", 1)[0]

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                        "round_id": round_idx,
                                        "prompt": cur_prompt,
                                        "text": outputs,
                                        "options": options,
                                        "option_char": cur_option_char,
                                        "answer_id": ans_id,
                                        "model_id": "CogVLM",
                                        "prompt_detail": qs,
                                        "metadata": {}}) + "\n")
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", type=str, default="mmaad_aad_base")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--prompt_id", default=0, type=int)
    args = parser.parse_args()

    eval_model(args)
