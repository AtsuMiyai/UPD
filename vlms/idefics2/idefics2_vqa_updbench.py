import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
import math
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
from PIL import Image
from io import BytesIO
import base64
from datasets import load_dataset


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
    questions = load_dataset("MM-UPD/MM-UPD", name=args.data_name, split="test")
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b", trust_remote_code=True)
    model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b", device_map="cuda", trust_remote_code=True).eval()

    for row in tqdm(questions, total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            eval_type = row['type']
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

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": qs},
                    {"type": "image"},
                ],
            }]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(images=image, text=text, return_tensors="pt").to('cuda')

            with torch.inference_mode():
                generated_text = model.generate(**inputs, max_new_tokens=500)
                outputs = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
                start_index = outputs.find("Assistant: ")
                generated_output = outputs[start_index + len("Assistant: "):]

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "eval_type": eval_type,
                                       "round_id": round_idx,
                                       "prompt": cur_prompt,
                                       "text": generated_output,
                                       "options": options,
                                       "option_char": cur_option_char,
                                       "answer_id": ans_id,
                                       "model_id": "Idefics2",
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
