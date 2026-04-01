import torch
import argparse
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


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

Your answer:
{answer}"""


prompt_template_noExplain = """You are an AI assistant that predicts whether the given code was written by a large language model (LLM) or by a student.
The code was submitted by a student for an assignment in an introductory Python programming course.
The problem description below is the assignment given to the students.

Please output a single line containing only 0 or 1:
- Output 1 if the code is likely written by an LLM.
- Output 0 if the code is likely written by the student.

Assignment:
{problem}

Code submitted by a student:
{code}

Your answer:
{answer}"""


def get_formatting_prompts_func(model_name, template_name):
    if template_name == "explain":
        overall_temp = prompt_template
    else:
        print("use prompt_template_noExplain")
        overall_temp = prompt_template_noExplain

    if model_name == "meta-llama/CodeLlama-7b-Instruct-hf":
        response_temp = "\nYour answer:"  # For CodeLlama, \n is needed. Without it, tokenizer.tokenize make "Your answer:" into  ['▁Your', '▁answer', ':']
    else:
        response_temp = "Your answer:"

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["id"])):
            label = example["label"][i]
            if template_name == "explain":
                reason_for_prediction = example["reason_for_prediction"][i]
                answer = f"{label}\n{reason_for_prediction}"
            else:
                answer = label

            text = overall_temp.format(
                problem=example["problem"][i], code=example["code"][i], answer=answer,
            )
            output_texts.append(text)
        return output_texts

    return formatting_prompts_func, response_temp


def main():
    # Argument parser for command line inputs
    parser = argparse.ArgumentParser(description="Fine-tune Llama Guard 3 with LoRA")

    # Add arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/CodeLlama-7b-Instruct-hf",
        help="Pretrained model name or path to local model",
    )
    parser.add_argument(
        "--template_name", type=str, default="explain", help="template name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/ep5",
        help="Where to save the fine-tuned model",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU/TPU core/CPU for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU/TPU core/CPU for evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="the number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=5, help="Number of epochs to train"
    )

    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("check tokenizer.pad_token:", tokenizer.pad_token)

    # LoRA configuration
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)

    # Load and preprocess dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/train.jsonl",
            "val": "data/val.jsonl",
            "test": "data/test.jsonl",
        },
    )

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        # evaluation_strategy="steps",
        evaluation_strategy="epoch",
        logging_steps=10,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=10_000,
        ddp_find_unused_parameters=False,  # For DistributedDataParallel (DDP)
        max_seq_length=1024,
    )

    formatting_prompts_func, response_template = get_formatting_prompts_func(
        model_name=args.model_name, template_name=args.template_name
    )
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )

    if args.model_name == "meta-llama/CodeLlama-7b-Instruct-hf":
        response_template_ids = response_template_ids[
            2:
        ]  # For codellama, with this code, the output is ['Your', '▁answer', ':']. without this code, the output is ['▁', '<0x0A>', 'Your', '▁answer', ':']

    print("model_name", args.model_name)
    print("response_template", response_template)
    print("response_template_ids", response_template_ids)

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
