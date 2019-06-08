import nltk, re
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup
#FOR WORKING WITH DIRECTORY CONTAING FILES
# import os
# for filename in os.listdir("D:\\ATML Project Data\\Raw data"):
#     if filename.endswith(".html") or filename.endswith(".htm"):
#         url = str(filename)

#url = 'file:///D:/ATML%20Project%20Data/Code/32006D0101_EN_NOT.html'
#url = 'file:///D:/ATML%20Project%20Data/Raw%20data/32003D0181_EN_NOT.html'
url = 'file:///D:/ATML%20Project%20Data/Raw%20data/21953A0704(01)_EN_NOT.html'
html = request.urlopen(url).read().decode('utf8')

#FILE CONTAING STOPWORD
f = open("StopWord.txt", "r")
StopWord = f.read().split()


#READING HTML FILE
raw = BeautifulSoup(html, 'html.parser')#.get_text()



#FINDING TITLE FROM HTML
#STILL NEEDS TO REMOVE TAGS, AAGE PEECHE SE
# a = raw.select('.rech')[0].text
Title = []
for elem in raw(text=re.compile(r'Title and reference')):
    x = elem.parent.find_next_siblings('strong')

    # for div in x.findAll('p'):
    #     Title.append(div.text.replace('\n', ''))
#print(children)
#print(x)
# print(len(Title))





#FOR FINDING LEBELS/descriptors FROM A FILE
descriptors = []
for elem in raw(text=re.compile(r'EUROVOC descriptor')):
    x = elem.parent.parent
    for div in x.findAll('a'):
        descriptors.append(div.text.replace('\n', ''))
print(descriptors)

#FOR FINDING TEXT DATA FROM HTML FILE
b = raw.select('.texte')
rawtextdata =  BeautifulSoup( str(b[0]) , 'html.parser').get_text()
tokens = word_tokenize(rawtextdata)

#FOR REMOVING STOPWORDS
wordsFiltered = []
for w in tokens:
    if w not in StopWord:
        wordsFiltered.append(w)
#print(wordsFiltered)
