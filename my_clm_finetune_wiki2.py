# Pre-trained model, LoRA, Train parts are referenced from https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_with_additional_tokens.ipynb
# Deepspeed, Distributed learning parts are referenced from https://github.com/microsoft/DeepSpeedExamples/blob/master/training/HelloDeepSpeed/train_bert_ds.py#L17

import os

os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="true"

import transformers
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import (  
    AutoModelForCausalLM,   
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch
from deepspeed.accelerator import get_accelerator
import numpy as np
from huggingface_hub import login

import my_load_dataset
from my_utils import my_ppl_metric
from ds_utils import log_dist, is_rank_0
from my_configs import ModelArguments, DataArguments, TrainingArguments
import random


#hf_access_token = '' # Set your 'Write' hf_access_token

# (0) Basic configuration setting

hf_parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
# Add my own configurations
hf_parser.add_argument("--my_finetune_method", type=str, default='FFT')
hf_parser.add_argument("--my_lora_dim", type=int, default=8)
hf_parser.add_argument("--my_lora_alpha", type=int, default=16)
hf_parser.add_argument("--my_lora_dropout", type=float, default=0.05)
hf_parser.add_argument("--my_max_length", type=int, default=512)

hf_parser.add_argument("--my_dataset_shuffle", type=int, default=0)

hf_parser.add_argument("--my_gradient_checkpointing", type=int, default=0)

model_args, data_args, training_args, my_args = hf_parser.parse_args_into_dataclasses()

# Seed fix for distributed learning
torch.manual_seed(training_args.seed)
np.random.seed(training_args.seed)
random.seed(training_args.seed)

device_rank = int(os.environ.get("RANK", -1))

device = (torch.device(get_accelerator().device_name(), device_rank) if (device_rank > -1)
              and get_accelerator().is_available() else torch.device("cpu"))


# (1) Load pre-trained model and tokenizer

# Example for additional tokens (including redefinitions of common specials, e.g., PAD, BOS, EOS)
# You should define new additional tokens at here,
# You need additional effort to make new embedding vectors be trainable in PEFT when using LoRA
# (WARNING) If you add additional tokens, it becomes hard to load the saved_checkpoint.
#my_additional_tokens = ['<|pad|>']

model_name = model_args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(
	    model_name,
        token=hf_access_token,
	    #pad_token=SpecialTokens.pad_token.value,
	    #bos_token=SpecialTokens.bos_token.value,
	    #eos_token=SpecialTokens.end_target.value,
	    #additional_special_tokens=SpecialTokens.list(),
	)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_access_token,
    #low_cpu_mem_usage=True
    # attn_implementation ="flash_attention_2", # leading to an error
)

# 'GPTNeoXForCausalLM' Model: "https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py"
# 'GPTNeoForCausalLM'  Model: "https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py"
pad_token_id = tokenizer.add_tokens('<|pad|>', special_tokens=True)
tokenizer.pad_token_id = pad_token_id # GPTNeoForCausalLM does not have padding token
tokenizer.pad_token = '<|pad|>'
model.resize_token_embeddings(len(tokenizer))


# (2) Apply Fine-tuning methods
# (Note) DeepSpeed looks efficient for FFT, but less efficient for LoRA

# List up weights in the pre-trained model
#weights = [ name for name, value in model.named_parameters() if 'weight' in name]
#print(weights)
if my_args.my_finetune_method == "LoRA":
    #lora_modules = ["embed_in", "query_key_value", "embed_out"] # for 'EleutherAI/polyglot-ko-1.3b'
    #lora_modules = ["wte", "k_proj", "v_proj", "q_proj", "lm_head"] # for 'EleutherAI/gpt-neo-1.3B'
    lora_modules = ["embed_tokens", "k_proj", "v_proj", "q_proj", "o_proj", "lm_head"] # for 'meta-llama/Llama-3.2-3B'

    log_dist("LoRA target modules : {}".format(lora_modules),
             ranks=[0])
    config = LoraConfig(
        r=my_args.my_lora_dim, lora_alpha=my_args.my_lora_alpha, lora_dropout=my_args.my_lora_dropout, target_modules=lora_modules
    )
    model = get_peft_model(model, config)

    log_dist(model.print_trainable_parameters(),
             ranks=[0])
elif my_args.my_finetune_method == "FFT":
    #set_trainable_list = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    #print("IMPORTANT WARNING: FFT is not making entire parameters to be .requires_grad=True")
    for n, p in model.named_parameters():
        p.requires_grad = True
    model.train()

    total_params = 0
    train_params = 0
    not_trainable_param_names = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_params += p.numel()
            train_params += p.numel()
            #trainable_param_names.append(name)
        else:
            total_params += p.numel()
            not_trainable_param_names.append(name)

    log_dist("{}/{} Trainable Param.".format(train_params, total_params), ranks=[0])
    #log_dist("Non-Trainable Layers: \n{}".format(not_trainable_param_names), ranks=[0])

if bool(my_args.my_gradient_checkpointing) is True:
    model.gradient_checkpointing_enable() # for memory efficiency (WARNING: it can slow down training)
    log_dist("(WARNING) Gradient Checkpointing Enable. It could slow down training", ranks=[0])


# (3) Load Dataset and Preprocessing
train_dataset, valid_dataset, test_dataset = my_load_dataset.Call_CLM_Wiki2_Datasets(tokenizer, maxlen=my_args.my_max_length)
training_args.label_names = ["labels"]

if bool(my_args.my_dataset_shuffle) is True:
    log_dist("Dataset Shuffle Enabled (Seed : {})".format(training_args.seed), ranks=[0])
    train_dataset.shuffle(seed=training_args.seed)
    valid_dataset.shuffle(seed=training_args.seed)
    test_dataset.shuffle(seed=training_args.seed)


# (4) Define Trainer

callbacks = []

# Define custom metric for PPL
#training_args.metric_for_best_model = "eval_ppl"
#training_args.greater_is_better = False
#training_args.batch_eval_metrics = True
#compute_metrics = my_ppl_metric # None
compute_metrics = None

# Because I set tokenizer, data_collator is automatically set to 'DataCollatorWithPadding'. It will add padding tokens for each minibatch
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator= DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), # DataCollator automatically copies 'input_ids' as 'labels' and mark -100 for paddings
    callbacks=callbacks,
    compute_metrics=compute_metrics,
)

# Note that ZeRO2 optimization does not support evaluations, Use ZeRO3 for evaluations
# But ZeRO2 supports validation.

#initial_test_results = trainer.evaluate(eval_dataset=test_dataset)

trainer.train()

#final_test_results = trainer.evaluate(eval_dataset=test_dataset)

#print("Initial Test Results : \n", initial_test_results)
#print("Final Test Results : \n", final_test_results)

