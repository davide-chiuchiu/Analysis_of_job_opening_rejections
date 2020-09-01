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
import string

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