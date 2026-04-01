gpu=$1
LLM_model=$2

CUDA_VISIBLE_DEVICES=$gpu python det-eval.py -m $LLM_model

python LLM-as-a-judge.py -i ./evaluation/result_$LLM_model.jsonl -o ./evaluation/LLM_as_a_judge_$LLM_model.jsonl

