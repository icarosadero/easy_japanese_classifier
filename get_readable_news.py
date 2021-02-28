from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from tqdm import tqdm
import os
from crawler import crawl_article, read
from nn import classify
import numpy as np
import argparse
import urllib.parse

parser = argparse.ArgumentParser()
parser.add_argument('-s',type=int, help="Ideal number of paragraphs.")
parser.add_argument('-t',type=float, help="Score threshold. Must be within [0,1].")
parser.add_argument('-a', type=float, help="Multiplicative factor for weight. The smaller this number, the more the number of paragraphs will diverge from the ideal number.")
args = parser.parse_args()

assert os.path.exists("model.torch")
assert os.path.exists("word_index.pickle")
assert os.path.exists("tag_index.pickle")

homepages = read("homepages.txt")

s = args.s if args.s else 10
t = args.t if args.t else 0.5
a = args.t if args.t else 0.0007

def is_absolute(url):
    return bool(urllib.parse.urlparse(url).netloc)

def weight(p,s=s,a=a):
    """
    Gaussian function for weighting scores.
    exp(-(p-s)**2)
    """
    return np.exp(-a*(p-s)*(p-s))

def get_links(home):
    request = requests.get(home)
    links = []
    if request.ok:
        soup = bs(request.text, features="html.parser")
        for a in soup.find_all("a", href=lambda x: (x is not None)):
            l = a.attrs['href']
            if not is_absolute(l):
                l = urllib.parse.urljoin(home,l)
            links.append(l)
        soup = bs(request.text, features="xml")
        for link in soup.find_all("link"):
            l = link.text
            if is_absolute(l):
                links.append(l)
    return links

def classify_article(link,positive_tag=1):
    paragraphs = crawl_article(link)
    scores = list(map(classify,paragraphs))
    positive_count = len(list(filter(lambda x: x[0]==positive_tag,scores)))
    N = positive_count/len(scores) if len(scores) !=0 else 0 #Ratio of paragraphs with positive tag.
    total_score = weight(len(paragraphs))*N
    return total_score

def get_positive_links(home,positive_tag=1,threshold=.7):
    pos_links = []
    for link in tqdm(get_links(home), desc="Classifying articles"):
        score = classify_article(link,positive_tag)
        if  score > threshold:
            pos_links.append(link)
    return pos_links

if __name__=="__main__":
    for page in homepages:
        print("Current page:", page)
        for link in get_positive_links(page):
            print(link)
            
