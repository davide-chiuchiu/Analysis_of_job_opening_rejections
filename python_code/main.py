#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 11:34:39 2020

@author: dabol99

This script initializes the nlp analysis of the emails where I got job opening
rejections. To this end, it loads the email in the downloaded_emails folder 
(not included in the git repository for privacy concerns).
"""

# import all the relevant libraries 
import matplotlib.pyplot
import nltk
import numpy
import os 
import pandas
import scipy
import seaborn
import sklearn


# set current work directory to the one with this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import functions from auxiliary files
from parse_one_email import parse_email
from text_utilities import preprocess_sender_info_for_nltk

# get list of files within the downloaded_emails folder
downloaded_emails_path = os.path.dirname(os.getcwd()) + '/downloaded_emails'
list_of_eml_files = [file for file in os.listdir(downloaded_emails_path) if file.endswith('.eml')]

# parse all emails and store their bodies and metadata in one dataframe
emails = [parse_email(downloaded_emails_path + '/' + email_name) for email_name in list_of_eml_files]
dataframe_emails = pandas.DataFrame(emails).reindex(columns = ['Date', 'From', 'Subject', 'Body'])

"""
Clean email dataset
"""
# strip quoted text from emails and linkedin random text
# based on "From:"
dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('From:.+', '')
# based on "Från: "
dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('Från:.+', '')
# based on linkedIn "Personal information Name "
dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('Personal information Name .+', '')
# based on linkedIn "View Message ©"
dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('View Message ©.+', '')

# get raw email and store it into a new field
dataframe_emails['Sender_email'] = dataframe_emails['From'].str.extract(pat = '([\+\w\.-]+@[\w\.-]+)')

"""
group mail together based on From information
"""
# remove stopwords and punctuation symbols from text
dataframe_emails['processed_From'] = dataframe_emails.apply(lambda x: preprocess_sender_info_for_nltk(x['From']) , axis = 1)

# create tfidf embedding of processed_From
corpus = dataframe_emails['processed_From'].tolist()
tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,2))
tfidf_vectorizer.fit(corpus)
embedded_processed_From = tfidf_vectorizer.transform(corpus)

# create dendrogram from dense representation of embedded_processed_From
hierarchical_clustering = scipy.cluster.hierarchy.linkage(embedded_processed_From.toarray(), metric = 'cosine', method = 'complete')
dendrogram = scipy.cluster.hierarchy.dendrogram(hierarchical_clustering)

# cluster email senders based on distance in dendrogram
clustering_strategy = sklearn.cluster.AgglomerativeClustering(n_clusters = None, compute_full_tree = True, distance_threshold =  0.2, affinity = 'cosine', linkage = 'complete')
clustering_strategy.fit(embedded_processed_From.toarray())

dataframe_emails['grouped_From'] = clustering_strategy.labels_




"""
Analysis of the cleaned dataset
"""
# count number of sentences in each email and show the distribution
dataframe_emails['Sentences count'] = dataframe_emails['Body'].apply(lambda x: len(nltk.tokenize.sent_tokenize(x)))
plot_lenght_distribution, (ax_box, ax_hist) = matplotlib.pyplot.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
seaborn.boxplot(dataframe_emails['Sentences count'], ax = ax_box)
seaborn.distplot(dataframe_emails['Sentences count'], kde = False, bins = numpy.arange(0.5, 30.5, 1), ax =ax_hist)
plot_destination_file = os.path.dirname(os.getcwd()) + '/Latex_summary_report/message_length_distribution.eps'
plot_lenght_distribution.savefig(plot_destination_file, format='eps')



