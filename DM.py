import nltk, re
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
#FOR WORKING WITH DIRECTORY CONTAING FILES
import os
import scipy.sparse


def tokens(x):
    return x.split(',')


i=0
corpus = []
for filename in os.listdir("D:\\ATML Project Data\\RandomTrain"):
    if filename.endswith(".html") or filename.endswith(".htm"):
              url = str('file:///D:/ATML Project Data/RandomTrain/' + filename)
              # print(url)
              html = request.urlopen(url).read().decode('utf8')
              raw = BeautifulSoup(html, 'html.parser')
              descriptors = []
              for elem in raw(text=re.compile(r'EUROVOC descriptor')):
                  x = elem.parent.parent
                  for div in x.findAll('a'):
                              descriptors.append(div.text.replace('\n', ''))

              c = ",".join(descriptors)
              corpus.append(c)

vectorizer = CountVectorizer(tokenizer=tokens, binary=True)
Y = vectorizer.fit_transform(corpus)

# print(vectorizer.get_feature_names())
# print(Y)
# print(Y.shape)
# print(type(corpus))

scipy.sparse.save_npz('D:\\ATML Project Data\\Matrices\\Y_Train.npz', Y)
