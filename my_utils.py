import torch
from torch.nn import CrossEntropyLoss


# This function is unstable
def my_ppl_metric(pred, compute_result=True):
    logits = pred.predictions[:,:-1,:].contiguous()
    labels = pred.label_ids[:,1:].contiguous()
    mask = torch.ne(labels, -100).type(torch.int)

    nll_fct = CrossEntropyLoss(reduction="none")
    nll = nll_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    ppl = torch.exp(nll)
    ppl = ppl * mask.reshape(-1)
    ppl = (ppl.sum() / mask.reshape(-1).sum()).item()

    #ppls = []
    #for b in range(logits.size(0)):
    #    for t in range(logits.size(1)):
    #        if labels[b,t] < 0:
    #            ppls.append(torch.exp(logits[b,t,labels[b,t]]).item())
    #return { "ppl": torch.tensor(ppls).mean().item() }
    return { "ppl": ppl }

'''
B = 3
T = 4
C = 5

logits = torch.randn((B, T, C))
labels = torch.tensor([[0,1,2,3],
                       [1,2,3,-100],
                       [2,3,4,-100]])

print(logits)
print(labels)

ppls = []
for b in range(logits.size(0)):
    for t in range(logits.size(1)):
        if labels[b,t] != -100:
            #ppls.append(torch.exp(logits[b,t,labels[b,t]]))
            ppls.append(logits[b,t,labels[b,t]])


print(ppls)
print(torch.tensor(ppls).mean().item())
'''
