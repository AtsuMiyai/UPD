# Based on https://github.com/Luodian/Otter/blob/main/pipeline/benchmarks/models/gpt4v.py

import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
import requests
import base64
import io
import time
from abc import ABC, abstractmethod
from io import BytesIO
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


def get_pil_image(raw_image_data) -> Image.Image:
    if isinstance(raw_image_data, Image.Image):
        return raw_image_data

    elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
        return Image.open(io.BytesIO(raw_image_data["bytes"]))

    elif isinstance(raw_image_data, str):  # Assuming this is a base64 encoded string
        image_bytes = base64.b64decode(raw_image_data)
        return Image.open(io.BytesIO(image_bytes))

    else:
        raise ValueError("Unsupported image data format")


class BaseModel(ABC):
    def __init__(self, model_name: str, model_path: str, *, max_batch_size: int = 1):
        self.name = model_name
        self.model_path = model_path
        self.max_batch_size = max_batch_size

    @abstractmethod
    def generate(self, **kwargs):
        pass

    @abstractmethod
    def eval_forward(self, **kwargs):
        pass


class OpenAIGPT4Vision(BaseModel):
    def __init__(self, api_key: str, model_name: str, max_new_tokens: int = 256):
        super().__init__("openai-gpt4", "gpt-4-vision-preview")
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

    @staticmethod
    def encode_image_to_base64(raw_image_data) -> str:
        if isinstance(raw_image_data, Image.Image):
            buffered = io.BytesIO()
            raw_image_data.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        raise ValueError("The input image data must be a PIL.Image.Image")

    def generate(self, text_prompt: str, raw_image_data):
        raw_image_data = get_pil_image(raw_image_data).convert("RGB")
        base64_image = self.encode_image_to_base64(raw_image_data)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
        }

        retry = True
        retry_times = 0
        while retry and retry_times < 5:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            else:
                print(f"Failed to connect to OpenAI API: {response.status_code} - {response.text}. Retrying...")
                time.sleep(10)
                retry_times += 1
        return "Failed to connect to OpenAI GPT4V API"

    def eval_forward(self, **kwargs):
        return super().eval_forward(**kwargs)


def read_jsonl(file_path):
    with open(file_path, 'r') as json_file:
        return [json.loads(line) for line in json_file]


def eval_model(args):
    questions = load_dataset("MM-UPD/MM-UPD", name=args.data_name)["test"]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    api_key = args.openai_api_key

    model = OpenAIGPT4Vision(api_key, args.model_name)

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
            eval_type = row['type']
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

            prompt = qs

            with torch.inference_mode():
                outputs = model.generate(prompt, image)

            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "eval_type": eval_type,
                                       "round_id": round_idx,
                                       "prompt": cur_prompt,
                                       "text": outputs,
                                       "options": options,
                                       "option_char": cur_option_char,
                                       "answer_id": ans_id,
                                       "model_id": args.model_name,
                                       "prompt_detail": prompt,
                                       "metadata": {}}) + "\n")
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", type=str, default="mmaad_aad_base")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="gpt-4-vision-preview")
    parser.add_argument("--openai-api-key", type=str, default='')
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--prompt_id", default=0, type=int)
    args = parser.parse_args()

    eval_model(args)
