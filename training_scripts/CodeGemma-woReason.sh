gpu="0,1,2,3"
output_dir="./output/CodeGemma-woReason"
lr=2e-5
num_train_epochs=30
model_name="google/codegemma-7b-it"

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --num_processes 4 --multi_gpu finetune_detector.py \
  --model_name $model_name \
  --output_dir $output_dir \
  --template_name woExplain \
  --learning_rate $lr \
  --num_train_epochs $num_train_epochs \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 