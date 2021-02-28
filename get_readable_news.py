from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from tqdm import tqdm
import os
from crawler import crawl_article
from nn import classify

assert os.path.exists("model.torch")
assert os.path.exists("word_index.pickle")
assert os.path.exists("tag_index.pickle")

homepage = "http://easyjapanese.net/news/normal/all?hl=en-US"

def get_links(home):
    request = requests.get(home)
    links = []
    if request.ok:
        soup = bs(request.text, features="html.parser")
        for a in soup.find_all("a", href=lambda x: (x is not None) and (("article" in x) or ("details") in x) ):
            links.append(a.attrs['href'])
    return links

def classify_article(link,positive_tag=1):
    paragraphs = crawl_article(link)
    scores = list(map(classify,paragraphs))
    positive_count = len(list(filter(lambda x: x==positive_tag,scores)))
    return positive_count/len(scores) if len(scores) !=0 else 0

def get_positive_links(home,positive_tag=1,threshold=.7):
    pos_links = []
    for link in get_links(home):
        if classify_article(link,positive_tag) > threshold:
            pos_links.append(link)
    return pos_links

if __name__=="__main__":
    for link in get_positive_links(homepage):
        print(link)
            
