from PIL.Image import new
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizerFast
import math
from torchvision import models
import torchvision
from utils import *
import os
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda:' +str(sys.argv[6]) if torch.cuda.is_available() else 'cpu')

def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)

def get_hidden_states(output, token_ids_word, i):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
        Select only those subword token outputs that belong to our word of interest
        and average them."""
    word_tokens_output = output[i, token_ids_word, :]
    return word_tokens_output.mean(dim=1)

def get_word_vector(sent, idx, tokenizer, output, i):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
        that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    return get_hidden_states(output, token_ids_word, i)


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
        self.bert = BertModel.from_pretrained('onlplab/alephbert-base').to(device)

    def forward(self, x):
        embeds = self.tokenizer(x, return_tensors='pt', padding=True)
        embeds = embeds.to(device)
        embeds = self.bert(**embeds)
        embeds = embeds.last_hidden_state

        all_embeddings = []
        for i, sent in enumerate(x):
            words_embedding = []
            words = sent.split(" ")
            for w in words:
                idx = get_word_idx(sent, w)
                words_embedding.append(get_word_vector(sent, idx, self.tokenizer, embeds, i))
            all_embeddings.append(torch.stack(words_embedding).permute(1, 0, 2))
        x = torch.stack(all_embeddings).squeeze(1)
        return x

class my_resnet(nn.Module):
    def __init__(self):
        super(my_resnet, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        self.modules = list(self.resnet101.children())[:-1]
        self.resnet101 = nn.Sequential(*self.modules)

    def forward(self, x_imgs):
        output = self.resnet101(x_imgs)
        return output.squeeze(2).squeeze(2)

class my_Model(nn.Module):
    def __init__(self, output_size):
        super(my_Model, self).__init__()
        self.LM_model = Bert()
        self.resnet = my_resnet()
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=3, 
                            batch_first=True)

        #self.fc1 = nn.Linear(2816, 2816)
        self.fc1 = nn.Linear(2816, output_size)
        self.output_size = output_size

    def get_words_up_to_j(self, sentences, j):
        new_sentences = []
        for sen in sentences:
            new_sentences.append(" ".join(sen.split()[:j]))
        return new_sentences

    def forward(self, img_vecs, sents, lens_sen):
        img_vecs = self.resnet(img_vecs)
        
        if max(lens_sen) == 1:
            loop_num = max(lens_sen)
            predictions = torch.zeros((img_vecs.shape[0], max(lens_sen), self.output_size)).to(device)
        else:
            loop_num = max(lens_sen) - 1 
            predictions = torch.zeros((img_vecs.shape[0], max(lens_sen) - 1, self.output_size)).to(device)

        count = 0
        for i in range(loop_num):
            sen_vecs = self.get_words_up_to_j(sents, i + 1)
            sen_vecs = self.LM_model(sen_vecs)
            
            curr_lens = []
            for j in lens_sen:
                if j < i:
                    curr_lens.append(j)
                else:
                    curr_lens.append(i + 1)

            x = pack_padded_sequence(sen_vecs, curr_lens, batch_first=True, enforce_sorted=False)
            _, (hn, _) = self.lstm(x)

            hn = hn.permute(1, 0, 2)[:, -1, :]

            x = torch.cat((hn, img_vecs), dim=1)
            x = self.fc1(F.gelu(x))

            predictions[:, count, :] = x[:, :]
            count += 1

        return predictions
