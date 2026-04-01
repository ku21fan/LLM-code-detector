gpu="0,1,2,3"
output_dir="./output/DeepseekCoder"
lr=2e-5
num_train_epochs=40
model_name="deepseek-ai/deepseek-coder-6.7b-instruct"

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --num_processes 4 --multi_gpu finetune_detector.py \
  --model_name $model_name \
  --output_dir $output_dir \
  --learning_rate $lr \
  --num_train_epochs $num_train_epochs \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 