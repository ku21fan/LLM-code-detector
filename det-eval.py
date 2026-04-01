import os
import json
import time
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset


prompt_template = """You are an AI assistant that predicts whether the given code was written by a large language model (LLM) or by a student.
The code was submitted by a student for an assignment in an introductory Python programming course.
The problem description below is the assignment given to the students.

Please output a single line containing only 0 or 1:
- Output 1 if the code is likely written by an LLM.
- Output 0 if the code is likely written by the student.

In the subsequent line, please provide a comprehensive explanation for your prediction in 5 sentences, detailing up to 5 reasons for the prediction.

Assignment:
{problem}

Code submitted by a student:
{code}

Your answer:"""


def classify_label(predicted_label: str) -> int:
    label_lower = predicted_label.lower()
    llm_index = label_lower.find("llm")
    student_index = label_lower.find("student")

    if llm_index != -1 and student_index != -1:
        return 1 if llm_index < student_index else 0
    elif llm_index != -1:
        return 1
    else:
        return 0


def main(args):
    model_name = args.model_name
    model_dir = f"./output/{model_name}/"
    print(model_dir)

    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/train.jsonl",
            "val": "data/val.jsonl",
            "test": "data/test.jsonl",
        },
    )
    dataset = dataset["test"]
    # print(dataset)

    device = "cuda"
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, device_map=device
    )
    model = PeftModel.from_pretrained(base_model, model_dir)

    pad_token_id = tokenizer.eos_token_id

    # Set the model to evaluation mode
    model.eval()
    model.generation_config.do_sample = False

    prediction = {}
    output_path = f"./evaluation/result_{model_name}.jsonl"
    with open(output_path, "w") as keep_result:
        print("make empty file at:", output_path)

    with open(output_path, "a") as keep_result:
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            prompt = prompt_template.format(
                problem=sample["problem"], code=sample["code"]
            )
            input = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(
                **input,
                max_new_tokens=1024,
                pad_token_id=pad_token_id,
                use_cache=True,
                do_sample=False,
            )
            prompt_len = input["input_ids"].shape[-1]
            result = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
            result = result.strip().split("\n")  # .replace("\n", "(newline)")
            predicted_label = result[0]
            predicted_reason_for_prediction = "\n".join(result[1:])
            json_dict = {
                "id": sample["id"],
                "problem_name": sample["problem_name"],
                "problem": sample["problem"],
                "code": sample["code"],
                "label": sample["label"],
                "predicted_label": predicted_label,
                "predicted_reason_for_prediction": predicted_reason_for_prediction,
            }
            keep_result.write(json.dumps(json_dict, ensure_ascii=False) + "\n")

    """ detection evaluation """
    with open(output_path, "r") as f_result:
        results = f_result.readlines()

    print("*" * 80)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for idx, res in enumerate(results):
        res = json.loads(res)
        res_id = res["id"]
        label = int(res["label"])
        predicted_label = res["predicted_label"]
        try:
            tmp = predicted_label.split("\n")[0]
            predicted_label = int(tmp)
        except:
            predicted_label = classify_label(predicted_label)

        if predicted_label == 1 and label == 1:
            TP += 1
        elif predicted_label == 1 and label == 0:
            FP += 1
        elif predicted_label == 0 and label == 1:
            FN += 1
        elif predicted_label == 0 and label == 0:
            TN += 1

    acc = 0 if (TP + FP + TN + FN) == 0 else float(TP + TN) / (TP + FP + TN + FN) * 100
    precision = 0 if (TP + FP) == 0 else float(TP) / (TP + FP) * 100
    recall = 0 if (TP + FN) == 0 else float(TP) / (TP + FN) * 100
    Hmean = (
        0
        if (precision + recall) == 0
        else float(2 * precision * recall) / (precision + recall)
    )
    eval_result = f"{output_path}\tTP:{TP}\tFP:{FP}\tFN:{FN}\tTN:{TN}\n{acc:0.3f}\t{precision:0.3f}\t{recall:0.3f}\t{Hmean:0.3f}"
    print(eval_result)
    with open(f"./evaluation/all_eval_score.txt", "a") as all_eval:
        all_eval.write(f"{eval_result}\n")

    # collecting false positive
    with open(output_path, "r") as f_result:
        results = f_result.readlines()

    os.makedirs("./evaluation/false_positives/", exist_ok=True)
    false_positive_case = (
        f"./evaluation/false_positives/false_positive_{model_name}.txt"
    )
    with open(false_positive_case, "w") as fp:
        for idx, res in enumerate(results):
            res = json.loads(res)
            res_id = res["id"]
            label = int(res["label"])
            predicted_label = res["predicted_label"]
            try:
                tmp = predicted_label.split("\n")[0]
                predicted_label = int(tmp)
            except:
                predicted_label = classify_label(predicted_label)

            if predicted_label == 1 and label == 0:  # False positives
                fp.write("*" * 80 + "\n")
                fp.write(f"{output_path}\tidx:{idx+1}\tFP:{res_id}\n")
                fp.write(res["code"] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluation on student data")
    parser.add_argument("-m", "--model_name", help="model name")
    args = parser.parse_args()

    start_time = time.time()
    main(args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.0f} seconds")
