from collections import Counter
from datasets import load
from numpy.core.numeric import indices
import torch
import numpy as np
import _pickle as pickle
from torch.functional import split
from torch.nn.functional import max_pool1d
from torch.utils.data import DataLoader
from utils import *
from model import *
import random
from nltk.translate.bleu_score import sentence_bleu
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda:' + str(sys.argv[6]) if torch.cuda.is_available() else 'cpu')


def check(sen):
    sen = sen.strip().split(" ")
    if '[SEP]' in sen or len(sen) == max_len_sen:
        return True
    return False

def pad_batch(sentences):
    max_len = 0
    padded_sentences = []
    lens = []
    for sen in sentences:
        sen = sen.split()
        lens.append(len(sen))
        max_len = max(max_len, len(sen))
    
    for sen in sentences:
        sen = sen.split()
        if len(sen) < max_len:
            for i in range(max_len - len(sen)):
                sen.append('[PAD]')
        padded_sentences.append(' '.join(sen))

    return padded_sentences, lens

def train_epoch(train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:
        imgs = torch.stack(list(data[:, 0])).to(device)
        descriptions = list(data[:, 1])
        for i in range(5):
            sentences = [desc[i] for desc in descriptions] 
            pad_gold_sentences, lens = pad_batch(sentences)
            pred_sents = model(imgs, pad_gold_sentences, lens).permute(0, 2, 1)
            gold_sents = torch.as_tensor([[w2i[w] for w in sen.strip().split(" ")[1:]] for sen in pad_gold_sentences], dtype=torch.long).to(device)
            optimizer.zero_grad()
            loss = loss_function(pred_sents, gold_sents)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

    return total_loss / (len_dataset * 5)

def eval_epoch(loader):
    model.eval()
    bleu_1, bleu_2, bleu_3, bleu_4 = 0, 0, 0, 0
    for data in loader:
        imgs = torch.stack(list(data[:, 0])).to(device)
        descriptions = list(data[:, 1])[0]
        pred_sent = "[CLS]"

        while True:
            with torch.no_grad():
                    flag = check(pred_sent)
                    if flag == True:
                        break
                    pred_next_w = model(imgs, [pred_sent], [len(pred_sent.strip().split(" "))])
                    idx = pred_next_w[:, -1, :].argmax(dim=1)
                    pred_sent += " " + str(list(w2i.keys())[idx])

        pred_sent = pred_sent.strip().split(" ")[1:-1]
        descriptions = [sen.strip().split(" ")[1:-1] for sen in descriptions]

        bleu_1 += sentence_bleu(descriptions, pred_sent, weights=(1, 0, 0, 0))
        bleu_2 += sentence_bleu(descriptions, pred_sent, weights=(0.5, 0.5, 0, 0))
        bleu_3 += sentence_bleu(descriptions, pred_sent, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 += sentence_bleu(descriptions, pred_sent, weights=(0.25, 0.25, 0.25, 0.25))

    bleu_1 /= len_dataset
    bleu_2 /= len_dataset
    bleu_3 /= len_dataset
    bleu_4 /= len_dataset

    return bleu_1, bleu_2, bleu_3, bleu_4


def train_loop(trainloader, epochs):
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(trainloader)
        print("Loss: " + str(train_loss))
    torch.save(model.state_dict(), str(sys.argv[3]))

def collate_fn(data):  
    return np.array(data, dtype=object)

if __name__ == '__main__':   
    desc = pickle.load(open(str(sys.argv[2]), "rb"))

    loader_imgs = pickle.load(open(str(sys.argv[4]), "rb"))
    loader_desc = pickle.load(open(str(sys.argv[5]), "rb"))
    dataset = get_dataset(loader_imgs, loader_desc)

    vocab, max_len_sen = get_vocab(desc)
    w2i = {w: i for i, w in enumerate(vocab)}
    
    if str(sys.argv[1]) == "train":
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=collate_fn)
    
    len_dataset = len(loader.dataset)

    loss_function = nn.CrossEntropyLoss(ignore_index=w2i["[PAD]"]).to(device)

    model = my_Model(len(vocab))

    if str(sys.argv[1]) != "train":
        model.load_state_dict(torch.load(str(sys.argv[3])))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    model = model.to(device)

    if str(sys.argv[1]) == "train":
        train_loop(loader, 10)
        bleu_1, bleu_2, bleu_3, bleu_4 = eval_epoch(loader)
        print("BLEU_1" + str(bleu_1))
        print("BLEU_2" + str(bleu_2))
        print("BLEU_3" + str(bleu_3))
        print("BLEU_4" + str(bleu_4))
    else:
        bleu_1, bleu_2, bleu_3, bleu_4 = eval_epoch(loader)
        print("BLEU_1" + str(bleu_1))
        print("BLEU_2" + str(bleu_2))
        print("BLEU_3" + str(bleu_3))
        print("BLEU_4" + str(bleu_4))