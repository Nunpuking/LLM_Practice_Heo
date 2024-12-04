GPU_LIST=$1
WORLD_SIZE=$2
PORT=34321

FINETUNE_METHOD="FFT" # FFT, LoRA


echo "(Note) Total batch size = world_size * per_device_train_batch_size * gradient_accumulation_steps"
echo "(WARNING) zero2 offload ds config, gradient checkpointing, 8x1/8 train/eval batchsizes for 1GPU (8/8 default)"

export NCCL_SHM_DISABLE=1
# PT model list: EleutherAI/gpt-neo-1.3B, meta-llama/Llama-3.2-3B, mistralai/Mistral-7B-v0.1

#deepspeed --num_gpus=$WORLD_SIZE my_clm_finetune_wiki2.py \
deepspeed --include localhost:$GPU_LIST my_clm_finetune_wiki2.py \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --deepspeed ./deepspeed_configs/ds_zero2_offload_bf16.json \
    --data_path ./ \
    --output_dir ./my_clm_finetune_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --max_grad_norm 1.0 \
    --eval_steps 100 \
    --eval_on_start False \
    --load_best_model_at_end True \
    --my_dataset_shuffle 0 \
    --my_gradient_checkpointing 1 \
    --my_finetune_method $FINETUNE_METHOD \
    --report_to none \
    --seed 0 \
    --my_lora_dim 64 \
    --my_lora_alpha 128 \
    --my_lora_dropout 0.0
