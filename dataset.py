import torch
from torch.utils import data
from itertools import islice
import Bio.SeqIO
import csv

list_positive = []
list_negative = []

dict_positive = {}
dict_negative = {}

dict_positive_GC = {}
dict_negative_GC = {}

with open("data/positive_interaction.fa", 'r') as file:
    i = 1
    gene_name = ''
    for line in file:
        if i % 2 != 0:
            gene_name = line[1:-1]
        else:
            dict_positive[gene_name] = line[0:-1]
        i += 1

with open("data/positive_interaction.csv", 'r') as file:
    for line in islice(file, 1, None):
        tmp = line.split(",")
        interact = []
        interact.append(tmp[0] + ':' + tmp[1] + '-' + tmp[2])
        interact.append(tmp[3] + ':' + tmp[4] + '-' + tmp[5])
        if interact[0] not in dict_positive.keys() or interact[1] not in dict_positive.keys():
            continue
        list_positive.append(interact)


for x in Bio.SeqIO.parse('data/CTCF_peak.seq.fa', 'fasta'):
    dict_negative[x.id] = str(x.seq)

with open("data/newSameDistanceNa.txt", 'r') as file:
    for line in file:
        tmp = line.split("\t")
        interact = []
        interact.append(tmp[0])
        interact.append(tmp[1][:-1])
        if interact[0] not in dict_negative.keys() or interact[1] not in dict_negative.keys():
            continue
        list_negative.append(interact)


class myDataset(data.Dataset):
    def __init__(self, list_p, list_n, dict_p, dict_n):
        self.list_p = list_p
        self.list_n = list_n
        self.dict_p = dict_p
        self.dict_n = dict_n
    
    def getGCcount(self, seq):
        num = 0
        for i in seq:
            if i in ['G', 'g', 'C', 'c']:
                num += 1
        return float(num/len(seq))

    def getSeq(self, index):
        process_list = self.list_p if index < len(self.list_p) else self.list_n
        process_dict = self.dict_p if index < len(self.list_p) else self.dict_n
        index = index if index < len(self.list_p) else (index - len(self.list_p))
        interact = process_list[index]
        gene_seq = process_dict[interact[0]] + process_dict[interact[1]]
        return gene_seq

    def __getitem__(self, index):
        flag_label = True if index < len(self.list_p) else False
        process_list = self.list_p if index < len(self.list_p) else self.list_n
        process_dict = self.dict_p if index < len(self.list_p) else self.dict_n

        index = index if index < len(self.list_p) else (index - len(self.list_p))

        interact = process_list[index]
        gene_seq = process_dict[interact[0]] + process_dict[interact[1]]
        gc1, gc2 = self.getGCcount(process_dict[interact[0]]), self.getGCcount(process_dict[interact[1]])

        one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0],
                'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

        encode_list = []
        for i, element in enumerate(gene_seq, 0):
            # if i in one_hot.keys():
            gcv = gc1 if i < len(process_dict[interact[0]]) else gc2
            encode_list.append(one_hot[element])

        seq = torch.tensor(encode_list, dtype=torch.float).t().unsqueeze(dim=0)
        label = torch.tensor(1, dtype=torch.int64) if flag_label else torch.tensor(0, dtype=torch.int64)
        return seq, label

    def __len__(self):
        return len(self.list_p) + len(self.list_n)


dataset = myDataset(list_positive, list_negative, dict_positive, dict_negative)

train_test_ratio = 0.8
n_train = int(len(dataset) * train_test_ratio)
n_val = int(len(dataset) * 0.1)
n_test = len(dataset) - n_train - n_val
trainset, valset, testset = data.random_split(dataset, [n_train, n_val, n_test])

train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=512,
    shuffle=True
)

val_loader = data.DataLoader(
    dataset=valset,
    batch_size=512,
    shuffle=True
)

test_loader = data.DataLoader(
    dataset=testset,
    batch_size=512,
    shuffle=True
)

def tensor2dict(seq):
    preSeq = ''
    one_hot = {'[1.0, 0.0, 0.0, 0.0]': 'A', '[0.0, 1.0, 0.0, 0.0]': 'C', '[0.0, 0.0, 1.0, 0.0]': 'G', '[0.0, 0.0, 0.0, 1.0]': 'T', '[0.0, 0.0, 0.0, 0.0]': 'N'}
    seq = seq.squeeze(dim=0).t().numpy().tolist()
    for i in range(len(seq)):
        preSeq += one_hot[str(seq[i])]
    return preSeq

lines = [('seq', 'label')]
for seq, label in trainset:
    preSeq = tensor2dict(seq)
    lines.append((preSeq, label.item()))
with open('train_loader.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(lines)

lines = [('seq', 'label')]
for seq, label in valset:
    preSeq = tensor2dict(seq)
    lines.append((preSeq, label.item()))
with open('val_loader.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(lines)

lines = [('seq', 'label')]
for seq, label in testset:
    preSeq = tensor2dict(seq)
    lines.append((preSeq, label.item()))
with open('test_loader.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(lines)

if __name__ == '__main__':
    # print(len(dataset))
    for i in range(len(dataset)):
        s1, s2 = dataset[i]
        # print(s1.shape, s2.shape)   # 4000
        print(s1)
        print(s2)
        print(s2.item())
        if i == 6:
            break
    # print(len(trainset), len(valset))   # 172032   43008
    # print(len(dataset))   # 215040

    # for seq, label in val_loader:
    #     print(seq.shape, label.shape)
    #     print("**" * 30)
    #     print(seq)
    #     print("**" * 30)
    #     print(label)
    #     break
