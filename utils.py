from typing import Counter
from numpy.lib.type_check import imag
import pandas as pd
import numpy as np
import string
from torch.utils.data import Dataset
import os
import subprocess
import pickle
import time
from collections import Counter
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
#from deep_translator import GoogleTranslator
import torch


def load_text(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    text = text.split('\n')
    # close the file
    file.close()
    return text

def get_splits():
    train_imgs = load_text("image_caption/text_folder/Flickr_8k.trainImages.txt")
    test_imgs = load_text("image_caption/text_folder/Flickr_8k.testImages.txt")
    dev_imgs = load_text("image_caption/text_folder/Flickr_8k.devImages.txt")
    train_imgs = [x for x in train_imgs if x != '']
    test_imgs = [x for x in test_imgs if x != '']
    dev_imgs = [x for x in dev_imgs if x != '']

    return train_imgs, dev_imgs, test_imgs

def load_descriptions(filename):
    descriptions = load_text(filename)
    mapping = dict()
    # process lines
    for line in descriptions:
        # split line by white space
        tokens = line.split()
        if len(tokens) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # remove filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        # add 'startseq' and 'endseq' to each description
        image_desc = ' '.join(image_desc)
        # store the first description for each image
        if image_id not in mapping.keys():
            mapping[image_id] = []
        mapping[image_id].append(image_desc)
    return mapping

def clean_description(desc_dict):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    count = 0
    start_time = time.time()
    counter = 0
    for key, desc_list in desc_dict.items():
        #if counter == 10:
        #    break
        for i in range(len(desc_list)):
            count += 1
            if count % 500 == 0:
                print("--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
            desc = desc_list[i]
            desc = GoogleTranslator(source='en', target='iw').translate(desc)
            # tokenize
            desc = desc.split()
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            desc = ' '.join(desc) + ' [SEP]'
            # store as string
            desc_list[i] = desc
        #counter += 1

def get_max_len(desc):
    max_len = 0
    for key in desc:
        for description in desc[key]:
            description = description.split()
            max_len = max(max_len, len(description))
    return max_len

class MYDataset(Dataset):
    # load the dataset
    def __init__(self, X, Y):
        # store the inputs and outputs
        self.X = X
        self.y = Y

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def image_to_tensor(image_name):
    img = Image.open(image_name)
    img_bgr = preprocess(img)
    return img_bgr

def create_img_vecs():
    train_imgs, dev_imgs, test_imgs = get_splits()
    img_vectors_trn = []
    img_vectors_val = []
    img_vectors_tst = []
    for img in train_imgs:
        img_vectors_trn.append(image_to_tensor("image_caption/Flicker8k_Dataset/" + img))

    for img1, img2 in zip(dev_imgs, test_imgs):
         img_vectors_val.append(image_to_tensor("image_caption/Flicker8k_Dataset/" + img1))
         img_vectors_tst.append(image_to_tensor("image_caption/Flicker8k_Dataset/" + img2))

    return img_vectors_trn, img_vectors_val, img_vectors_tst

def get_desc():
    desc_dict = load_descriptions("image_caption/text_folder/Flickr8k.token.txt")
    clean_description(desc_dict)
    dataMatrix = [desc_dict[i] for i in desc_dict.keys()]
    return dataMatrix

def get_vocab(dataMatrix):
    my_words = Counter()
    max_len_sen = 0
    for caps in dataMatrix:
        for sen in caps:
            max_len_sen = max(max_len_sen, len(sen.split()))
            my_words.update(Counter(sen.split()))
    my_words["[PAD]"] = 1
    del(my_words["[CLS]"])
    return list(my_words.keys()), max_len_sen

def get_dataset(img_vectors, dataMatrix):
    my_dataset = MYDataset(img_vectors, dataMatrix)
    return my_dataset