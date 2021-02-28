# easy_japanese_classifier

This program can filter out articles that you can read from the website `https://www3.nhk.or.jp/news/easy` based on examples you provide. It is itended for people who are learning Japanese.

If you change the tokenizer used on `nn.py` as well as the links in the project, you can adapt the code to work with some other language in some other website.

### In case you are Elias (ñ sabe ler)

Gets easy to read articles from a website.

### Prerequisites

- Python3
- pip
- PyTorch
- Beautiful Soup 4

### Installing
Clone the repository to your local computer:

`https://github.com/icarosadero/easy_japanese_classifier.git`

This script uses neubig's NLP Deep Continous Bag of Words model:

`git clone https://github.com/neubig/nn4nlp-code.git`

On `nn.py` on line 2, make sure that the path inside of `sys.path.append` is pointing to the `model.py` model from neubig's repository.

Install the remaining packages:

```
pip install 'konoha[all]'
pip install beautifulsoup4
pip install pytorch
pip install tqdm
```

Get this file from the konoha repository in particular and paste it with the rest of the project:

`wget https://github.com/himkt/konoha/raw/master/data/model.spm`

On `nn.py` on line 17, make sure that `model_path` is pointing to this file.

## Usage

First, get a list of article links in Japanese that you are able to read and paste it on `positive_links.txt`, one on each line. Do the same for Japanese articles that you are not able to read on `negative_links.txt`. It is recommended that the total number of sentences from these articles to be about the same. After that, paste the urls of the sites you wanto to scrape in `homepages.txt`.

Then run the scripts in the following order:

```
python3 crawler.py
python3 nn.py
python3 get_readable_news.py
```

After the first time you run those scripts, you will only need to re-execute `crawler.py` and `nn.py` if  you update either `positive_links.txt` or `negative_links.txt`.

You can pass some optional arguments to `get_readable_news.py` for fine tuning. You can see them by executing:

```
python3 get_readable_news.py -h
```
.

## Authors

* **Ícaro Lorran Lopes Costa** - (https://www.linkedin.com/in/icarolorran/)

