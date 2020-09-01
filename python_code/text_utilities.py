#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:16:18 2020

@author: dabol99

This files contains functions that are useful to process strings for NLTK 
applications. 
"""

# import modules
import nltk
import scipy
import sklearn
import string

"""
This function wraps the pre-processing and clustering operations to perform on
the From field in the dataframe_emails. 
"""
def cluster_emails_by_From(dataframe_emails):
    # create corpus by removing removing stopwords and punctuation symbols from text
    corpus = dataframe_emails.apply(lambda x: preprocess_sender_info_for_nltk(x['From']) , axis = 1).tolist()

    # embed corpus as tfidf matrix    
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,2))
    tfidf_vectorizer.fit(corpus)
    embedded_processed_From = tfidf_vectorizer.transform(corpus).toarray()
    
    # create dendrogram from dense representation of embedded_processed_From
    hierarchical_clustering = scipy.cluster.hierarchy.linkage(embedded_processed_From, metric = 'cosine', method = 'complete')
    scipy.cluster.hierarchy.dendrogram(hierarchical_clustering)
    
    # cluster email senders based on distance in dendrogram
    clustering_strategy = sklearn.cluster.AgglomerativeClustering(n_clusters = None, compute_full_tree = True, distance_threshold =  0.2, affinity = 'cosine', linkage = 'complete')
    clustering_strategy.fit(embedded_processed_From)

    return clustering_strategy.labels_
    



"""
This function performs tokenize the input string text while it removes all 
stopwords and punctuation symbols.
"""
def preprocess_sender_info_for_nltk(text):
    
    # define set with stopwords and punctuation signs.
    stopset = nltk.corpus.stopwords.words('english') + list(string.punctuation) + list(['"', '``', "''"])

    # tokenize text (replacement of . into )
    tokenized_text = nltk.tokenize.word_tokenize(text.lower().replace('.', ' '))

    # remove stopwords from tokenized text
    tokenized_text_without_stopwords = " ".join([word for word in tokenized_text if not word in stopset])

    return tokenized_text_without_stopwords