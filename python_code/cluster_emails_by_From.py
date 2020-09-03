#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:23:53 2020

@author: dabol99
This file contains the function that performs the clustering of the emails based
on their From field
"""
import sklearn
import scipy

from text_utilities import build_tfidf_embedding_from_dataframe




"""
This function clusters the emails in dataframe emails based on the 'From' field.
Clustering is performed by building a hierarchical tree with distances specified
by metric and method, and then cutting it at cut_distance. Words that has to be
removed from the clustering can be specified via extra_tokens_to_remove. 
build_dendrogram specifies if the algorithm has to build a visible dendrogram 
of the hierarchical tree
"""
def cluster_emails_by_From(dataframe_emails, cut_distance, metric = 'cosine', method = 'complete', extra_tokens_to_remove = None, build_dendrogram = True):
    # create tfidf_embedded_corpus_From after removing stopwords, 
    # punctuation symbols and extra_tokens_to_remove
    tfidf_embedded_corpus_From, _ = build_tfidf_embedding_from_dataframe(dataframe_emails, 'From', extra_tokens_to_remove = extra_tokens_to_remove)
    
    # cast as dense matrix due to requirements of hierarchical clustering functions
    dense_tfidf_embedded_corpus_From  = tfidf_embedded_corpus_From.toarray()

    # cluster email senders based on distance in dendrogram - uses default n-gram range
    clustering_strategy = sklearn.cluster.AgglomerativeClustering(n_clusters = None, compute_full_tree = True, distance_threshold =  cut_distance, affinity = metric, linkage = method)
    clustering_strategy.fit(dense_tfidf_embedded_corpus_From) # requires dense matrix

    # create dendrogram from dense representation of embedded_processed_From
    if build_dendrogram == True:
        hierarchical_clustering = scipy.cluster.hierarchy.linkage(dense_tfidf_embedded_corpus_From, metric = metric, method = method)
        scipy.cluster.hierarchy.dendrogram(hierarchical_clustering)
    
    return clustering_strategy.labels_