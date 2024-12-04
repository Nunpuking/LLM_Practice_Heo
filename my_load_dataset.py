import torch
import copy
import json
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from ds_utils import log_dist

def Call_CLM_AIHUB_Datasets(data_path, tokenizer):

    class MyCustomDataset(Dataset):
        def __init__(self, model_inputs):
            self.model_inputs = model_inputs

        def __len__(self):
            return len(self.model_inputs["input_ids"])

        def __getitem__(self, idx):
            return { "input_ids": self.model_inputs["input_ids"][idx],\
		     "attention_mask": self.model_inputs["attention_mask"][idx]\
		}

    # Preprocessing with unit of 1000 samples
    def preprocess_function(examples):
        batch_size = len(examples)
        model_inputs = tokenizer(examples) #, padding=True)\
        for i in range(batch_size):
	    #print(i)
	    #print(model_inputs["input_ids"][i])
	    #print(tokenizer.decode(model_inputs["input_ids"][i]))
            sample_input_ids = model_inputs["input_ids"][i] + [tokenizer.eos_token_id]
            model_inputs["input_ids"][i] = sample_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        return MyCustomDataset(model_inputs) 
    with open(data_path, "r") as read_json:
        dataset = json.load(read_json)

    train_dataset = preprocess_function(dataset["train"])
    log_dist("# of Trainset : {}".format(len(train_dataset)), ranks=[0])
    valid_dataset = preprocess_function(dataset["valid"])
    log_dist("# of Valildset : {}".format(len(valid_dataset)), ranks=[0])
    test_dataset = preprocess_function(dataset["test"])
    log_dist("# of Testset : {}".format(len(test_dataset)), ranks=[0])

    return train_dataset, valid_dataset, test_dataset

   
def Call_CLM_Wiki2_Datasets(tokenizer, test_mode=False, maxlen=None):
    
    # Preprocessing with unit of 1000 samples
    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        model_inputs = tokenizer(examples[text_column]) #, padding=True, max_length=maxlen-2) # -2 for bos, eos
        #print(batch_size, len(model_inputs["input_ids"]))
        for i in range(batch_size):
            if len(model_inputs["input_ids"][i]) > 0:
                if model_inputs["input_ids"][i][0] == tokenizer.bos_token_id:
                    sample_input_ids = model_inputs["input_ids"][i]
                else:
                    sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][i]
                if model_inputs["input_ids"][i][-1] == tokenizer.eos_token_id:
                    sample_input_ids = sample_input_ids
                else:
                    sample_input_ids = sample_input_ids + [tokenizer.eos_token_id]
            else:
                sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][i] + [tokenizer.eos_token_id]

            model_inputs["input_ids"][i] = sample_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        #model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"]) # Note that if the element of labels is -100, it is ignored during loss computation.
        #del model_inputs["token_type_ids"] # For PolyglotKO model
        #model_inputs["labels"] = model_inputs["input_ids"]
        return model_inputs

    # Preprocessing with unit of 1000 samples
    def test_preprocess_function(examples):
        batch_size = len(examples[text_column])
        model_inputs = tokenizer(examples[text_column]) #, padding="longest", max_length=maxlen)
        for i in range(batch_size):
            if len(model_inputs["input_ids"][i]) > 0:
                if model_inputs["input_ids"][i][0] == tokenizer.bos_token_id:
                    sample_input_ids = model_inputs["input_ids"][i]
                else:
                    sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][i]
                if model_inputs["input_ids"][i][-1] == tokenizer.eos_token_id:
                    sample_input_ids = sample_input_ids
                else:
                    sample_input_ids = sample_input_ids + [tokenizer.eos_token_id]
            else:
                sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][i] + [tokenizer.eos_token_id]
            model_inputs["input_ids"][i] = sample_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

	#model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"]) # Note that if the element of labels is -100, it is ignored during loss computation.
        #del model_inputs["token_type_ids"] # For PolyglotKO model
        model_inputs["labels"] = model_inputs["input_ids"]
        return model_inputs

    def remove_emptylines(processed_dataset):
	# remain only if data contains only EOS token
        new_dataset = processed_dataset.select(
                (
			    i for i in range(len(processed_dataset))
			    if len(processed_dataset[i]["input_ids"]) > 2 # 2 for bos, eos
			)
		    )
        return new_dataset

    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    text_column = "text"
    label_column = "text"

    # For large dataset, save and load cache file please.
    # 'remove_columns' removes original raw data column after preprocessing
    preprocess_function_instance = preprocess_function if test_mode is False else test_preprocess_function
    processed_datasets = dataset.map(
	preprocess_function_instance,
	batched=True,
	num_proc=1,
	remove_columns=dataset["train"].column_names,
	load_from_cache_file=False,
	desc="Running tokenizer on dataset",
    )

    if test_mode is False:
        train_dataset = remove_emptylines(processed_datasets["train"])
        log_dist("# of Trainset : {}".format(len(train_dataset)), ranks=[0])
        valid_dataset = remove_emptylines(processed_datasets["validation"])
        log_dist("# of Validset : {}".format(len(valid_dataset)), ranks=[0])
        test_dataset = remove_emptylines(processed_datasets["test"])
        log_dist("# of Testset : {}".format(len(test_dataset)), ranks=[0])
        return train_dataset, valid_dataset, test_dataset
    else:
        test_dataset = remove_emptylines(processed_datasets["test"])
        log_dist("# of Testset : {}".format(len(test_dataset)), ranks=[0])
        return test_dataset

