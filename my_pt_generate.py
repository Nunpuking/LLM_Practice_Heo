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



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# (1) Load pre-trained model and tokenizer
model_path = "EleutherAI/polyglot-ko-1.3b"
pt_model_dir = None #"./AIHUB_Results/CLM_PolyglotKo_LoRA_AIHUB" # None for PT model test

#print(model_config)

# Model class source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
model = AutoModelForCausalLM.from_pretrained(model_path)

if pt_model_dir is not None:
    tokenizer = AutoTokenizer.from_pretrained(pt_model_dir)
    model = PeftModel.from_pretrained(model, pt_model_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()

tokenizer.padding_side = "left"

print(model)

#context = ["Effects of Castor Meal on the Growth"] #Performance and Carcass Characteristics of Beef Cattle"]

#from test_samples import sentences
from train_samples import sentences

for i in range(len(sentences)):
    context = [sentences[i]]

    model_inputs = tokenizer(context, padding=True)
    print(model_inputs)
    model_input = torch.tensor(model_inputs["input_ids"]).to(device)

    output_token = model.generate(
            model_input,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            )

    print("Input context: ", context)
    target_predicted = tokenizer.decode(output_token[0], skip_special_tokens=False)
    print("Generated: ", target_predicted)
