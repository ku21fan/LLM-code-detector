import argparse
import asyncio
import json
import os
from collections import defaultdict

import base64
import aiohttp
import tqdm

import time


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


async def get_response(session, content: str, max_tokens: int):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY') }",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-2024-11-20",
        "messages": [{"role": "user", "content": content},],
        "temperature": 1.0,  # for diversity
        "max_tokens": max_tokens,
    }

    while True:
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 429:
                    await asyncio.sleep(NUM_SECONDS_TO_SLEEP)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)


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


async def main(args):
    async with aiohttp.ClientSession() as session:

        input_path = open(os.path.expanduser(args.input_path))

        if os.path.isfile(os.path.expanduser(args.output_path)):
            cur_responses = [
                json.loads(line) for line in open(os.path.expanduser(args.output_path))
            ]
        else:
            cur_responses = []

        output_file = open(f"{args.output_path}", "w")

        tasks = []
        cur_js_list = []
        for idx, jsonline in tqdm.tqdm(enumerate(input_path)):
            line = json.loads(jsonline)
            try:
                data_id = line["id"]
                problem_name = line["problem_name"]
                problem = line["problem"]
                code = line["code"]
                label = line["label"]
            except:
                assert idx, jsonline

            cur_js = {
                "id": data_id,
                "problem_name": problem_name,
                "problem": problem,
                "code": code,
                "label": label,
            }

            content = prompt_template.format(problem=problem, code=code)

            if idx >= len(cur_responses):
                task = asyncio.create_task(
                    get_response(session, content, args.max_tokens)
                )
                tasks.append(task)
                cur_js_list.append(cur_js)
            else:
                print(f"Skipping {idx} as we already have it.")

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Process results and write to file as before
        for result, cur_js in zip(results, cur_js_list):
            # Assuming `cur_js` is prepared as before:
            result = result.strip().split("\n")  # .replace("\n", "(newline)")
            cur_js["predicted_label"] = result[0]
            cur_js["predicted_reason_for_prediction"] = "\n".join(result[1:])
            output_file.write(json.dumps(cur_js, ensure_ascii=False) + "\n")
            output_file.flush()

        output_file.close()

    """ detection evaluation """
    model_name = "gpt-4o"
    output_path = args.output_path
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
    parser.add_argument("-i", "--input_path")
    parser.add_argument("-o", "--output_path")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    start_time = time.time()
    asyncio.run(main(args))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.0f} seconds")
