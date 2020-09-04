#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:16:18 2020

@author: dabol99

This files contains functions that are useful to preprocess strings for nltk 
and to build document embeddings.
"""

# import modules
import nltk
import re
import sklearn
import sklearn.cluster
import string




"""
This wrapper builds the tfidf_embedded_corpus matrix with the corresponding 
embedding_labels from the collection of strings stored at column_label in
dataframe. The function preprocess the  collection of strings by removing stopwords,
punctuation and extra_tokens_to_remove
"""
def build_tfidf_embedding_from_dataframe(dataframe, column_label, extra_tokens_to_remove = None, ngram_range = (1,1), remove_numbers = True):
    # create corpus by removing removing stopwords, punctuation symbols and extra_tokens_to_remove
    corpus = dataframe.apply(lambda x: preprocess_corpus(x[column_label], extra_tokens_to_remove, remove_numbers = remove_numbers) , axis = 1).tolist()
    
    # embed corpus as tfidf matrix    
    tfidf_embedded_corpus, embedding_labels = build_tfidf_embedding_from_corpus(corpus, ngram_range = ngram_range)

    return tfidf_embedded_corpus, embedding_labels




"""
This function performs tokenize the input string text while it removes all 
stopwords and punctuation symbols.
"""
def preprocess_corpus(text, extra_tokens_to_remove = None, remove_numbers = True):
    # remove numbers from text
    if remove_numbers == True:
        text = re.sub('[0-9]', '', text)
    
    # create stopwords and punctuation signs to remove based on english stopwords,
    # punctuatio and extra_tokens_to_remove
    if extra_tokens_to_remove == None:
        stopset = nltk.corpus.stopwords.words('english') + list(string.punctuation)
    else:
        stopset = nltk.corpus.stopwords.words('english') + list(string.punctuation) + extra_tokens_to_remove    

    # tokenize text (replacement of . into )
    tokenized_text = nltk.tokenize.word_tokenize(text.lower().replace('.', ' '))

    # remove stopwords from tokenized text
    stemmer = nltk.stem.lancaster.LancasterStemmer()
    tokenized_text_without_stopwords = " ".join([stemmer.stem(word) for word in tokenized_text if not word in stopset])

    return tokenized_text_without_stopwords




"""
This function performs the tfidf embedding of a corpus using ngrams in ngram_range
and then it returns the tfidf matrix of the corpus.
"""
def build_tfidf_embedding_from_corpus(corpus, ngram_range = (1,1)):
    # embed corpus as tfidf matrix    
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range = ngram_range)
    tfidf_vectorizer.fit(corpus)
    
    # extract tfidf embedding as sparse matrix, and extract embedding labels
    embedded_corpus = tfidf_vectorizer.transform(corpus)
    embedding_labels = tfidf_vectorizer.get_feature_names()
    
    return embedded_corpus, embedding_labels