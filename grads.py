import numpy as np
import Bio.SeqIO
import torch
import random
from captum.attr import DeepLift, Saliency, DeepLiftShap
import sys
import csv
# sys.path.append('..')

valset_list = []
with open("val_loader.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        valset_list.append(row)
valset_list.pop(0)

torch.manual_seed(12301)
a = torch.randperm(len(valset_list)).numpy().tolist()
index = random.sample(a, 5000) 

device = torch.device('cuda:3')
model = torch.load("model_yjw.pkl", map_location='cuda:3')

# device = torch.device('cuda:0')
# model = Net()
# model.load_state_dict(torch.load('../model_yjw.pkl'))
# model = model.to(device)


# dict_seq = {}
# for x in Bio.SeqIO.parse(r'../../../data/gene_6000.fasta', 'fasta'):
#     dict_seq[x.id.split('_')[0]] = str(x.seq)

# # 4943
# val_gene = []
# with open(r'../res/list_val_gene.txt', 'r') as f:
#     for line in f:
#         val_gene.append(line[:-1])


# def seq2tensor(seq, device):
#     seq = seq.upper()
#     one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0],
#                 'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}
#     res = []
#     for i in seq:
#         res.append(one_hot[i])
#     return torch.tensor(res, dtype=torch.float, requires_grad=True, device=device).t().unsqueeze(dim=0)

def toTensor(seq):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0],
                'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}
    encode_list = []
    for element in seq:
        encode_list.append(one_hot[element])
    encode_seq = torch.tensor(encode_list, dtype=torch.float).t().unsqueeze(dim=0)
    return encode_seq

list_seq, list_hyp_grad, list_grad = [], [], []
for i in range(len(index)):
    seq, label = valset_list[index[i]]
    seq_tensor = toTensor(seq).unsqueeze(dim=0).to(device)
    label = torch.tensor(int(label), dtype=torch.int64).to(device)

    seq_tensor.requires_grad = False
    outs = model(seq_tensor)
    _, pred = torch.max(outs.data, 1)

    if pred.item() == label.item():
        seq_tensor.requires_grad = True
        model.zero_grad()
        # dl = DeepLift(model, multiply_by_inputs=False)
        # grads = dl.attribute(inputs=seq_tensor, target=label.item())
        dls = DeepLiftShap(model, multiply_by_inputs=False)
        grads = dls.attribute(inputs=seq_tensor, target=label.item(), baselines=torch.zeros(10, 1, 4, 4000).to(device))
        
        list_hyp_grad.append(grads[0].squeeze(dim=0).detach().cpu().t().numpy().tolist())
        list_grad.append((grads * seq_tensor).detach().cpu().squeeze(dim=0).squeeze(dim=0).t().numpy().tolist())
        list_seq.append(seq.upper())

list_hyp_grad, list_grad = np.array(list_hyp_grad), np.array(list_grad)

# Mean-normalize the hupothetical contributions at each base
list_hyp_grad = (list_hyp_grad - np.mean(list_hyp_grad, axis=-1)[:, :, None])

np.save('seqs_dls.npy', np.array(list_seq))

import h5py
f = h5py.File("scores_dls.h5", 'w')
g = f.create_group("contrib_scores")
g.create_dataset("task0", data=list_grad)
g = f.create_group("hyp_contrib_scores")
g.create_dataset("task0", data=list_hyp_grad)
f.close()












