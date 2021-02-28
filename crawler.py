from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from tqdm import tqdm

def read(file_path):
    lines = []
    with open(file_path) as f:
        for l in f:
            l = l.strip()
            if l[0]!="#":
                lines.append(l)
    return lines

def crawl_article(link):
    request = requests.get(link)
    paragraphs = []
    if request.ok:
        soup = bs(request.text, features="html.parser")
        paragraphs = map(lambda x: x.get_text(),soup.find_all("p"))
    return list(paragraphs)

def generate_dataset(file, category_marker):
    links = read(file)
    paragraphs = []
    
    for link in tqdm(links):
        paragraphs += crawl_article(link)
    data = list(map(lambda x: {"p":x, "c":category_marker},paragraphs))
    return data

if __name__=="__main__":
    positive_data = generate_dataset("positive_links.txt", "1")
    negative_data = generate_dataset("negative_links.txt", "0")
    total = positive_data + negative_data
    df = pd.DataFrame(total)
    df = df.sample(frac=1)
    df.to_csv("data.csv")
