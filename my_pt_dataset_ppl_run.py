# Heavily referenced from https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_with_additional_tokens.ipynb

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="true"
print("WARNING: CUDA_VISIBLE_DEVICES is mannually set to '0'")

import transformers
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
    DataCollatorForLanguageModeling,
)
import torch
from dataclasses import dataclass, field
from typing import Optional
from dataclass_csv import DataclassReader
from torch.utils.data import Dataset, DataLoader

import my_load_dataset

import math
import copy
import numpy as np
from enum import Enum

from tqdm import tqdm


# (0) Test configurations
max_length = 512
print("(WARNING) Maxlen is set to {}".format(max_length))


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# (1) Load pre-trained model and tokenizer
task = 'Wiki2' # AIHUB, Wiki2

data_path = './aihub_datasets/clm_aihub_agr_data.json'

# PT model list: EleutherAI/gpt-neo-1.3B, meta-llama/Llama-3.2-3B, mistralai/Mistral-7B-v0.1
model_path = "meta-llama/Llama-3.2-3B" #"EleutherAI/polyglot-ko-1.3b"
pt_model_dir = "./my_clm_finetune_results/checkpoint-9180" #"./WIKI2_Results/CLM_PolyglotKo_LoRA_Wiki2_DS" # None for PT model test

#print(model_config)

# Model class source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
model = AutoModelForCausalLM.from_pretrained(model_path)

if pt_model_dir is not None:
    tokenizer = AutoTokenizer.from_pretrained(pt_model_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

pad_token_id = tokenizer.add_tokens('<|pad|>', special_tokens=True)
tokenizer.pad_token_id = pad_token_id # GPTNeoForCausalLM does not have padding token
tokenizer.pad_token = '<|pad|>'
model.resize_token_embeddings(len(tokenizer))
if pt_model_dir is not None:
    model = PeftModel.from_pretrained(model, pt_model_dir)
model.to(device)


# (2) Load Dataset and Preprocessing
if task == 'Wiki2':
    #print("IMPORTANT WARNING: loading validset for testset")
    test_dataset = my_load_dataset.Call_CLM_Wiki2_Datasets(tokenizer, test_mode=True, maxlen=max_length)
elif task == 'AIHUB':
    train_dataset, _, test_dataset = my_load_dataset.Call_CLM_AIHUB_Datasets(data_path, tokenizer)



#encodings = test_dataset.features
'''
print("WARNING: This is non-fixed length PPL computation (sentence-wise)")
print("WARNING: This is sentence-by-sentence PPL computation (not in a mini-batch)")
total_len = len(test_dataset)
ppls = []
losses = []
for idx, sample in enumerate(test_dataset):
    print("{}/{} processed".format(idx+1, total_len), end="\r")
    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)
    target_ids = torch.tensor(sample["labels"]).unsqueeze(0).to(device)
    seq_len = input_ids.size(1)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
        neg_log_likelihood = outputs.loss
    losses.append(neg_log_likelihood.mean())
    ppls.append(torch.exp(neg_log_likelihood).mean())

loss = torch.tensor(losses).mean()
print("Test Loss: {}".format(loss))
ppl = torch.tensor(ppls).mean()
print("Test PPL : {}".format(ppl))
'''

stride = max_length
total_len = len(test_dataset)
print("WARNING: This is fixed length PPL computation with {} stride".format(stride))
print("WARNING: This is sentence-by-sentence PPL computation (not in a mini-batch)")
ppls = []
for idx, sample in enumerate(test_dataset):
    print("{}/{} processed".format(idx+1, total_len), end="\r")
    
    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)
    target_ids = torch.tensor(sample["labels"]).unsqueeze(0).to(device)
    seq_len = input_ids.size(1)
   
    iterations = seq_len // stride + 1
    prev_end_loc = 0
    for i in range(iterations):
        end_loc = min(stride*(i+1), seq_len)
        tmp_input_ids = input_ids[:,prev_end_loc:end_loc]
        tmp_attention_mask = attention_mask[:,prev_end_loc:end_loc]
        tmp_target_ids = target_ids[:,prev_end_loc:end_loc]

        prev_end_loc = end_loc

        if tmp_input_ids.size(1) > 1:
            with torch.no_grad():
                outputs = model(tmp_input_ids, attention_mask=tmp_attention_mask, labels=tmp_target_ids)
                neg_log_likelihood = outputs.loss
            ppls.append(torch.exp(neg_log_likelihood).mean())

ppl = torch.tensor(ppls).mean()
print("Test PPL: {}".format(ppl))
