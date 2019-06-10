import scipy.sparse
from skmultilearn.adapt import MLkNN
import nltk, re
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import scipy.sparse


Y_train = scipy.sparse.load_npz('D:\\ATML Project Data\\Matrices\\Y_Train.npz')
X_train = scipy.sparse.load_npz('D:\\ATML Project Data\\Matrices\\X_Train.npz')



#FILE CONTAING STOPWORD
f = open("StopWord.txt", "r")
StopWord = f.read().split()

#TOKENIZER, FOR SPLITTING AT ','
def tokens(x):
    return x.split(',')

corpus = []
corpustext = []

#FOR WORKING WITH DIRECTORY CONTAING FILES
#Take notice of url (and paths) that you give as vairable, these are local dependent
for filename in os.listdir("D:\\ATML Project Data\\RandomTest"):
    if filename.endswith(".html") or filename.endswith(".htm"):
              url = str('file:///D:/ATML Project Data/RandomTest/' + filename)
              html = request.urlopen(url).read().decode('utf8')
              raw = BeautifulSoup(html, 'html.parser')


             #FOR FINDING LABELS/descriptors FROM A FILE
              descriptors = []
              for elem in raw(text=re.compile(r'EUROVOC descriptor')):
                  x = elem.parent.parent
                  for div in x.findAll('a'):
                              descriptors.append(div.text.replace('\n', ''))

              c = ",".join(descriptors)
              corpus.append(c)


              #FOR FINDING TEXT DATA FROM HTML FILE
              b = raw.select('.texte')
              rawtextdata =  BeautifulSoup( str(b[0]) , 'html.parser').get_text()
              tokens = word_tokenize(rawtextdata)
              #FOR REMOVING STOPWORDS
              wordsFiltered = []
              for w in tokens:
                  if w not in StopWord:
                      wordsFiltered.append(w)

              cc = " ".join(wordsFiltered)
              corpustext.append(cc)


#print(corpus)

# FOR MAKING SPARSE DOC-DESCRIPTOR MATRIX. IT'S A BINARY PRESENCE ABSENCE MATRIX
vectorizer1 = CountVectorizer(tokenizer=tokens, binary=True)
Y_test = vectorizer1.fit_transform(corpus)
# print(vectorizer.get_feature_names())
# print(Y)
# print(Y.shape)


# #FOR MAKING SPARSE DOC-TERM MATRIX. WITH TF-TDF WEIGHTING
vectorizer = TfidfVectorizer()
X_test = vectorizer.fit_transform(corpustext)
# print(vectorizer.get_feature_names())
# print(X)
# print(X.shape)


# # #FOR MAKING SPARSE DOC-TERM MATRIX. WITH Term frequency WEIGHTING
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpustext)
# print(vectorizer.get_feature_names())
# print(X)
# print(X.shape)



# print(X_test)
# print(X_test.shape)



classifier = MLkNN(k=1)
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)

print(Y_test)
print(Y_test.shape)
print(predictions)
print(predictions.shape)
