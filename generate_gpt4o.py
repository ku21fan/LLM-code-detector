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

# GEN_NUM_PER_PROBLEM = 400
# PROLBEM_LIST = ["ex1", "ex2", "ex3", "ex4", "ex5"]

GEN_NUM_PER_PROBLEM = 10
PROLBEM_LIST = ["ex1"]


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


async def main(args):
    async with aiohttp.ClientSession() as session:

        for problem_name in PROLBEM_LIST:
            output_path = f"data/raw/gpt_{problem_name}.txt"
            problem_file_path = f"{args.problem_dir}/{problem_name}.txt"
            print(
                f"problem_name: {problem_name}, problem_file_path: {problem_file_path}, output_path: {output_path}"
            )

            if os.path.isfile(os.path.expanduser(output_path)):
                cur_responses = [
                    json.loads(line) for line in open(os.path.expanduser(output_path))
                ]
            else:
                cur_responses = []

            response_file = open(f"{output_path}", "w")

            with open(problem_file_path, "r") as file:
                problem = file.read()

            content = (
                f"Write Python code without comments.\nSolve this problem: {problem}"
            )

            tasks = []
            cur_js_list = []
            for idx in tqdm.tqdm(range(GEN_NUM_PER_PROBLEM)):

                cur_js = {
                    "id": idx + 1,
                    "problem_name": problem_name,
                }

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
                cur_js["content"] = result
                response_file.write(json.dumps(cur_js, ensure_ascii=False) + "\n")
                response_file.flush()

            response_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-p", "--problem_dir", default="./data/problem")
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
