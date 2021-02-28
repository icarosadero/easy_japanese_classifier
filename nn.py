import sys
sys.path.append("nn4nlp-code/01-intro-pytorch")

import torch
from torch import nn
from torch.autograd import Variable
from model import DeepCBoW
import pickle
import pandas as pd
import random
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from konoha import WordTokenizer

konoha_tokenizer = WordTokenizer('Sentencepiece', model_path="/home/icaro/konoha_model.spm")

nlayers, emb_size, hid_size = 3, 6, 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w2i = defaultdict(lambda: len(w2i)) #word to index
w2i["<unk>"]
t2i = defaultdict(lambda: len(t2i)) #tag to index
def read_dataset(df):
    for i, row in df.iterrows():
        tag = row["c"]
        words = row["p"].lower().strip()
        yield ([w2i[str(x)] for x in konoha_tokenizer.tokenize(words)], t2i[tag])

def train(data_path):
    global w2i, t2i
    data = pd.read_csv(data_path)  
    train = list(read_dataset(data))
    w2i = defaultdict(lambda: w2i["<unk>"], w2i)
    nwords, ntags = len(w2i), len(t2i)
    
    model = DeepCBoW(nwords, ntags, nlayers, emb_size, hid_size)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for ITER in tqdm(list(range(100))):
        # Perform training
        random.shuffle(train)
        train_loss = 0.0
        for words, tag in train:
            words = torch.LongTensor(words).to(device)
            tag = torch.tensor([tag]).to(device)
            scores = model(words)
            loss = criterion(scores, tag)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    pickle.dump(dict(w2i),open("word_index.pickle","wb"))
    pickle.dump(dict(t2i), open("tag_index.pickle", "wb"))
    torch.save(model,"model.torch")

    return model

def classify(text):
    model = torch.load("model.torch")
    w2i = pickle.load(open("word_index.pickle","rb"))
    w2i = defaultdict(lambda: w2i["<unk>"] , w2i)
    t2i = pickle.load(open("tag_index.pickle", "rb"))
    i2t = {v:k for k,v in t2i.items()}
    words = text.lower().strip()
    input_ = torch.LongTensor([w2i[str(x)] for x in konoha_tokenizer.tokenize(words)]).to(device)
    cl = model(input_)[0].detach().cpu().numpy()
    cls = np.argmax(cl)
    return i2t[cls], cl

if __name__=="__main__":
    model = train("data.csv")
    

