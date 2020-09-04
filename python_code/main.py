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
import seaborn
import sklearn


# set current work directory to the one with this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import functions from auxiliary files
from build_email_dataframe import build_email_dataframe
from cluster_emails_by_From import cluster_emails_by_From
from text_utilities import build_tfidf_embedding_from_dataframe

# build dataframe of emails from database of stored emails
downloaded_emails_path = os.path.dirname(os.getcwd()) + '/downloaded_emails'
dataframe_emails = build_email_dataframe(downloaded_emails_path)

"""
label emails
"""
# group mail together based on From information 
# A 0.6 threshold for the three cut, use of 1-grams and the current extra
# tokens to remove do a decent (although improvable) job
extra_punctuation = ['"', '``', "''"]
email_domains = ['co', 'com', 'eu', 'io', 'it', 'net', 'se', 'uk']
buzzwords = ['teamtailor', 'email', 'mail', 'noreply', 'jobvite', 
             'no-reply', 'recruiting', 'Team', 'GmbH', 'lever', 
             'linkedin', 'people', 'careers', 'notification', 
             'system', 'successfactor']
extra_tokens_to_remove = extra_punctuation + email_domains + buzzwords
dataframe_emails['grouped_From'] = cluster_emails_by_From(dataframe_emails, 0.6, extra_tokens_to_remove = extra_tokens_to_remove)

for i in numpy.sort(dataframe_emails['grouped_From'].unique()):
    print(i)
    print(dataframe_emails[dataframe_emails['grouped_From'] == i]['From'])
    print('')

# lable emails by body content
# create tfidf embedding of the email bodies
tfidf_embedded_Bodies, embedding_Bodies_labels = build_tfidf_embedding_from_dataframe(dataframe_emails, 'Body', ngram_range = (1,2))

# inspect tfidf embedding
tfidf_dataframe = pandas.DataFrame(tfidf_embedded_Bodies.toarray(), columns = embedding_Bodies_labels)

# perform nmf decomposition
NMF_model = sklearn.decomposition.NMF(n_components = 3, init = 'nndsvd', random_state = 0)
pattern_coefficients = NMF_model.fit_transform(tfidf_embedded_Bodies)
linguistic_patterns = pandas.DataFrame(NMF_model.components_, columns = embedding_Bodies_labels).transpose()

# identify topics of linguistic patterns
for i in linguistic_patterns.columns:
    temp = linguistic_patterns[i].sort_values(ascending = False)
    print('pattern ' + str(i))
    print(temp.head(n = 10))
    print('')

    



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



