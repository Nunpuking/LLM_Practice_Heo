# referenced by https://huggingface.co/docs/transformers/perplexity

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import my_sentences

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


target_sentence = my_sentences.abnormal_ko_sentence
print("PPL target: ", target_sentence)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-1.3b")
print(model)
model = model.to(device)

encodings = tokenizer([my_sentences.normal_sentence, my_sentences.normal_ko_sentence], return_tensors="pt", padding=True)
print(encodings)


max_length = 512 # arbitrary selected (org: 'model.config.n_positions')
stride = 512
seq_len = encodings.input_ids.size(1)
print(seq_len)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    print("begin_loc: " ,begin_loc)
    end_loc = min(begin_loc + max_length, seq_len)
    print("end_loc: ", end_loc)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    print("trg_len: ", trg_len)
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    print("input_ids : \n", input_ids)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100
    print("target_ids : \n", target_ids)

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids) # GPTNeoXForCausalLM class automatically does next_word_prediction in forward method.


        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl)



