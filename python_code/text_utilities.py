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
import pandas

"""
this function uses nltk features to returns the word frequencies from the corpus
stored as a series of documents in corpus_as_series
"""
def compute_word_frequencies(corpus_as_series):
    corpus_condensed_as_string = corpus_as_series.str.cat(sep = ' ')
    tokenized_corpus = nltk.tokenize.word_tokenize(corpus_condensed_as_string)
    word_frequencies = nltk.probability.FreqDist(tokenized_corpus)
    return word_frequencies




"""
This function tokenizes and stems the input string text while it removes all 
stopwords and punctuation symbols. 
"""
def preprocess_corpus(text, extra_tokens_to_remove = None, remove_numbers = True):
    # remove numbers from text
    if remove_numbers == True:
        text = re.sub('[0-9]', '', text)
    
    # remove all special symbols from text
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)

    # create stopwords to remove based on english stopwords, and extra_tokens_to_remove
    if extra_tokens_to_remove == None:
        stopset = nltk.corpus.stopwords.words('english')
    else:
        stopset = nltk.corpus.stopwords.words('english') + extra_tokens_to_remove    

    # tokenize text (replacement of . into )
    tokenized_text = nltk.tokenize.word_tokenize(text.lower())

    # remove stopwords from tokenized text
    stemmer = nltk.stem.lancaster.LancasterStemmer()
    tokenized_text_without_stopwords = " ".join([stemmer.stem(word) for word in tokenized_text if not word in stopset])

    return tokenized_text_without_stopwords




"""
This wrapper builds the tfidf_embedded_corpus matrix with the corresponding 
embedding_labels and embedder object from the collection of strings stored 
at column_label in dataframe. The function preprocess the  collection of 
strings by removing stopwords, punctuation and extra_tokens_to_remove
"""
def build_tfidf_embedding_from_dataframe(dataframe, column_label, extra_tokens_to_remove = None, ngram_range = (1,1), remove_numbers = True):
    # create corpus by removing removing stopwords, punctuation symbols and extra_tokens_to_remove
    corpus = dataframe.apply(lambda x: preprocess_corpus(x[column_label], extra_tokens_to_remove, remove_numbers = remove_numbers) , axis = 1).tolist()
    
    # build tfidf vectorizer object from function  
    tfidf_vectorizer = build_tfidf_embedding_from_corpus(corpus, ngram_range = ngram_range)

    # extract tfidf embedding as sparse matrix, and extract embedding labels
    tfidf_embedded_corpus = tfidf_vectorizer.transform(corpus)
    embedding_labels = tfidf_vectorizer.get_feature_names()

    return tfidf_embedded_corpus, embedding_labels, tfidf_vectorizer





"""
This function performs the tfidf embedding of a corpus using ngrams in ngram_range
and then it returns the tfidf embedder object.
"""
def build_tfidf_embedding_from_corpus(corpus, ngram_range = (1,1)):
    # embed corpus as tfidf matrix    
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range = ngram_range)
    tfidf_vectorizer.fit(corpus)
    
    return tfidf_vectorizer




"""
This function uses the tfidf embedding of corpus to compute an automated 
list of buzzwords in the corpus. By definition, a word classify as buzzword if 
its ifd is in the buzzword_quantile_treshold quantile.
"""
def identify_custom_buzzwords(corpus, buzzword_quantile_treshold):
    # build tfidf embedding of corpus
    tfidf_corpus_embedding = build_tfidf_embedding_from_corpus(corpus, ngram_range = (1,1))
    # store tfidf vocabulary as dataframe with the corresponding idf
    tfidf_corpus_embedding_dataframe = pandas.DataFrame.from_dict(tfidf_corpus_embedding.vocabulary_, orient = 'index', columns = ['word index']).sort_values('word index')
    tfidf_corpus_embedding_dataframe['idf'] = tfidf_corpus_embedding.idf_

    # identify buzzwords as the ones in the buzzword_quantile_treshold quantile
    buzzword_idf_cutoff = tfidf_corpus_embedding_dataframe['idf'].quantile(buzzword_quantile_treshold)
    custom_buzzwords = sorted(tfidf_corpus_embedding_dataframe[tfidf_corpus_embedding_dataframe['idf'].between(0, buzzword_idf_cutoff)].index)
    
    # list of next possible buzzwords
    other_buzzword_idf_cutoff = tfidf_corpus_embedding_dataframe['idf'].quantile(2 * buzzword_quantile_treshold)
    other_custom_buzzwords = sorted(tfidf_corpus_embedding_dataframe[tfidf_corpus_embedding_dataframe['idf'].between(0, other_buzzword_idf_cutoff)].index)
    
    return custom_buzzwords, other_custom_buzzwords