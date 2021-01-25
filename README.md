# Multilabel classification 

Dataset: Euro-LEX (http://www.ke.tu-darmstadt.de/resources/eurlex)

Pre procession: 
  1. Tokenization
  2. Stop word removal
  3. Stemming
  
Feature Selection:
  1. TF-IDF
  2. TF

Models:
  1. Multi label K-nearest neighbor (ML-KNN)
  2. Binary Relevance(BR), using Naive Bayes classifiers.
  
Evaluation: K-Fold Cross Validation

Evaluation Metric: Micro-Average Precision and Recall.
