import argparse
import asyncio
import json
import os
from collections import defaultdict

import base64
import aiohttp
import tqdm

import time

NUM_SECONDS_TO_SLEEP = 0.5

prediction_prompt_reason4judge = """You are evaluating an explanation for a prediction task that determines whether a given code snippet was written by a student or by a large language model (LLM).
You are given a programming assignment, a code snippet written for that assignment, a predicted label ("student" or "LLM"), and an explanation for the prediction, consisting of multiple reasons.
Your task is to assess the quality of the explanation provided for such a prediction — not whether the prediction label itself is correct.

Please score the explanation based on:
1. Does it identify important and relevant features of the code in the context of the assignment?
2. Are the mentioned features valid and supported by the actual code?

Rate the explanation from 1 to 5:
- 5 = Excellent: Highlights key features with fully valid reasoning.
- 4 = Good: Mostly relevant and valid, with minor issues.
- 3 = Fair: Some relevant points, but also weak or unclear ones.
- 2 = Poor: Few valid points, mostly weak or irrelevant.
- 1 = Very poor: No clear or justifiable reasoning.

Just reply with the number only.
Assignment:
{problem}

Code snippet:
{code}

Predicted label:
{label}

Reasons for the prediction:
{reason}

Your score:"""


def classify_label(prediction: str) -> int:
    label_lower = prediction.lower()
    llm_index = label_lower.find("llm")
    student_index = label_lower.find("student")

    if llm_index != -1 and student_index != -1:
        return 1 if llm_index < student_index else 0
    elif llm_index != -1:
        return 1
    else:
        return 0


async def get_response(session, content: str, max_tokens: int):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY') }",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-2024-11-20",
        "messages": [{"role": "user", "content": content},],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 46,
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
                predicted_label = line["predicted_label"]
                predicted_reason_for_prediction = line[
                    "predicted_reason_for_prediction"
                ]
            except:
                assert idx, jsonline

            try:
                predicted_label = "LLM" if int(predicted_label) else "student"
            except:
                predicted_label = classify_label(
                    predicted_label + predicted_reason_for_prediction
                )

            cur_js = {
                "id": data_id,
                "problem_name": problem_name,
                "problem": problem,
                "code": code,
                "label": label,
                "predicted_label": predicted_label,
                "predicted_reason_for_prediction": predicted_reason_for_prediction,
            }

            content = prediction_prompt_reason4judge.format(
                problem=problem,
                code=code,
                label=predicted_label,
                reason=predicted_reason_for_prediction,
            )

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

        LLM_as_a_judge_score_list = []
        # Process results and write to file as before
        for result, cur_js in zip(results, cur_js_list):
            # Assuming `cur_js` is prepared as before:
            result = int(result)
            cur_js["LLM_as_a_judge"] = result
            LLM_as_a_judge_score_list.append(result)
            output_file.write(json.dumps(cur_js, ensure_ascii=False) + "\n")
            output_file.flush()

        output_file.close()

        average = sum(LLM_as_a_judge_score_list) / len(LLM_as_a_judge_score_list)
        tmp = f"{args.input_path}\tLLM_as_a_judge: {average}"
        print(tmp)
        with open(f"./evaluation/all_eval_score.txt", "a") as all_eval:
            all_eval.write(f"{tmp}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
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
