import nltk, re
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import scipy.sparse
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
import sklearn.metrics as metrics
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import KFold
import warnings


warnings.filterwarnings("ignore")

#FILE CONTAING STOPWORD
f = open("StopWord.txt", "r")
StopWord = f.read().split()

#FILE CONTAING Empty files
f1 =  open("BadFiles.txt", "r")
BadFiles = f1.read().split()


i=0
corpus = []
corpustext = []
stemmer = SnowballStemmer('english')



#TOKENIZER, FOR SPLITTING AT ','
def tokens(x):
    return x.split(',')


def TextTrain():
#FOR WORKING WITH DIRECTORY CONTAING FILES
#Take notice of url (and paths) that you give as vairable, these are local dependent
    for filename in os.listdir("D:\\ATML Project Data\\Raw data"):
        if (filename.endswith(".html") or filename.endswith(".htm")) and (filename not in BadFiles):
                  # print(filename)
                  url = str('file:///D:/ATML Project Data/Raw data/' + filename)
                  html = request.urlopen(url).read().decode('utf8')
                  raw = BeautifulSoup(html, 'html.parser')

                  #FOR FINDING TEXT DATA FROM HTML FILE
                  #USE TRY CATCH HERE
                  b = raw.select('.texte')
                  rawtextdata =  BeautifulSoup( str(b[0]) , 'html.parser').get_text()
                  tokens = word_tokenize(rawtextdata)
                  #FOR REMOVING STOPWORDS
                  wordsFiltered = []
                  for w in tokens:
                      if w not in StopWord:
                          w1 = stemmer.stem(w)
                          wordsFiltered.append(w1)


                  cc = " ".join(wordsFiltered)
                  corpustext.append(cc)


    # #FOR MAKING SPARSE DOC-TERM MATRIX. WITH TF-TDF WEIGHTING
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpustext)
    # print(vectorizer.get_feature_names())
    # print(X)
    # print(X.shape)


    # # #FOR MAKING SPARSE DOC-TERM MATRIX. WITH Term frequency WEIGHTING
    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(corpustext)
    # print(vectorizer.get_feature_names())
    # print(X)
    # print(X.shape)

    # scipy.sparse.save_npz('D:\\ATML Project Data\\Matrices\\X_Train.npz', X)
    # print('Hello')
    return(X)

def DescTrain():
#FOR WORKING WITH DIRECTORY CONTAING FILES
#Take notice of url (and paths) that you give as vairable, these are local dependent
    for filename in os.listdir("D:\\ATML Project Data\\Raw data"):
        if (filename.endswith(".html") or filename.endswith(".htm")) and (filename not in BadFiles) :
                  url = str('file:///D:/ATML Project Data/Raw data/' + filename)
                  html = request.urlopen(url).read().decode('utf8')
                  raw = BeautifulSoup(html, 'html.parser')

                   #FOR FINDING LEBELS/descriptors FROM A FILE
                  descriptors = []
                  for elem in raw(text=re.compile(r'EUROVOC descriptor')):
                      x = elem.parent.parent
                      for div in x.findAll('a'):
                          descriptors.append(div.text.replace('\n', ''))

                  c = ",".join(descriptors)
                  corpus.append(c)


    #print(corpus)

    # # FOR MAKING SPARSE DOC-DESCRIPTOR MATRIX. IT'S A BINARY PRESENCE ABSENCE MATRIX
    vectorizer = CountVectorizer(tokenizer=tokens, binary=True)
    Y = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names())
    # print(Y)
    # print(Y.shape)

    # scipy.sparse.save_npz('D:\\ATML Project Data\\Matrices\\Y.npz', Y)
    # print('Hi')
    return(Y)


def DataSplit(X,Y):

#---------------------Defining Classifiers-------------#
    # MLKNN CLASSIFICATION, WITH K=20
    # classifier = MLkNN(k=20)

    #BR
    # classifier = BinaryRelevance( classifier = SVC(), require_dense = [False, True])      #DOSENT WORK
    classifier = BinaryRelevance(GaussianNB())                                            #WORKS

    #CC, RandomForest
    # classifier = ClassifierChain(classifier = RandomForestClassifier(n_estimators=100), require_dense = [False, True])


    f1_micro = []
    f1_macro = []
    hamm = []
    precision_micro = []
    precision_macro = []
    recall_micro = []
    recall_macro = []

#----------------------------K Fold Cross Validation-----------------------#
    kf = KFold(n_splits=5, shuffle= True)
    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]        #train_test_split(X, Y, test_size=0.33)   #random_state=42

#     return(X_train, X_test, y_train, y_test)
#
# def Classification(X_train, X_test, y_train, y_test):

        classifier.fit(X_train, y_train)
        y_hat = classifier.predict(X_test)

        f1_micro.append( metrics.f1_score(y_test, y_hat, average='micro') )
        f1_macro.append( metrics.f1_score(y_test, y_hat, average='macro') )

        hamm.append( metrics.hamming_loss(y_test,y_hat) )

        recall_micro.append( metrics.recall_score(y_test, y_hat, average='micro') )
        recall_macro.append( metrics.recall_score(y_test, y_hat, average='macro') )

        precision_micro.append( metrics.precision_score(y_test, y_hat, average='micro') )
        precision_macro.append( metrics.precision_score(y_test, y_hat, average='macro') )

#------------------Avg-ing metric values------------------------------------------#

        # accuracy = metrics.accuracy_score(y_test, y_hat)

    print('Micro F1-score:', round( (sum(f1_micro))/(len(f1_micro)) , 3)  )
    # print(f1_micro)
    print('Macro F1-score:', round( (sum(f1_macro))/(len(f1_macro)) , 3) )
    print('Micro recall:', round( (sum(recall_micro))/(len(recall_micro)) , 3) )
    print('Macro recall:', round( (sum(recall_macro))/(len(recall_macro)) , 3) )
    print('Micro precision:', round( (sum(precision_micro))/(len(precision_micro)), 3) )
    print('Macro precision:', round( (sum(precision_macro))/(len(precision_macro)), 3) )
    print('Hamming Loss:', round( (sum(hamm))/(len(hamm)) , 3) )

        ## print('Accuracy:', (sum(f1_micro))/(len(f1_micro)) )
        ## print(metrics.classification_report(y_test, y_hat))
        ## print('Hamming Loss:', round(hamm,3) )


def main():
    X = TextTrain()
    Y = DescTrain()

    # print(Y.shape)
    #X_train, X_test, y_train, y_test =
    DataSplit(X,Y)
    # Classification(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
