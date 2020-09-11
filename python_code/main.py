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
import sklearn.ensemble
import imblearn


# set current work directory to the one with this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import functions from auxiliary files
from build_email_dataframe import build_email_dataframe
from cluster_emails_by_From import cluster_emails_by_From
from text_utilities import preprocess_corpus, find_keywords

# build dataframe of emails from database of stored emails. The building procedure 
# remove numbers, and it stems the email bodies after tokenization
downloaded_emails_path = os.path.dirname(os.getcwd()) + '/downloaded_emails'
dataframe_emails = build_email_dataframe(downloaded_emails_path)

"""
Prelimianry analysis
"""
# inspect imbalance between email types
Email_type_distribution = dataframe_emails['Email_type'].value_counts(normalize = False)

email_type_distribution_figure, email_type_distribution_axis = matplotlib.pyplot.subplots()
seaborn.countplot(x = 'Email_type', data = dataframe_emails, ax = email_type_distribution_axis)
plot_destination_file = os.path.dirname(os.getcwd()) + '/Latex_summary_report/email_type_distribution.eps'
email_type_distribution_figure.savefig(plot_destination_file, format='eps')

# count number of sentences in each email and show the distribution based on email types
dataframe_emails['Sentences count'] = dataframe_emails['Body'].apply(lambda x: len(nltk.tokenize.sent_tokenize(x)))

email_length_distribution_figure, email_length_distribution_axis = matplotlib.pyplot.subplots()
seaborn.boxplot(x = 'Sentences count', y = 'Email_type', data = dataframe_emails, ax = email_length_distribution_axis) 
plot_destination_file = os.path.dirname(os.getcwd()) + '/Latex_summary_report/message_length_distribution.eps'
email_length_distribution_figure.savefig(plot_destination_file, format='eps')

# find keywords based on word count
word_counter = sklearn.feature_extraction.text.CountVectorizer(stop_words = nltk.corpus.stopwords.words('english'), ngram_range=(1, 2))
grouped_dataframe_emails = dataframe_emails.groupby('Email_type')
list_common_words = grouped_dataframe_emails.apply(lambda x: find_keywords(x['Stemmed Body'], word_counter))

for i in list_common_words.index:
    print('Common words in ' + i + ' email type')
    print(list_common_words.loc[i])
    print('')




# """
# label emails
# """
# # group mail together based on From information 
# # A 0.6 threshold for the three cut, use of 1-grams and the current extra
# # tokens to remove do a decent (although improvable) job
# extra_punctuation = ['"', '``', "''"]
# email_domains = ['co', 'com', 'eu', 'io', 'it', 'net', 'se', 'uk']
# buzzwords = ['teamtailor', 'email', 'mail', 'noreply', 'jobvite', 
#              'no-reply', 'recruiting', 'Team', 'GmbH', 'lever', 
#              'linkedin', 'people', 'careers', 'notification', 
#              'system', 'successfactor']
# extra_tokens_to_remove = extra_punctuation + email_domains + buzzwords
# dataframe_emails['grouped_From'] = cluster_emails_by_From(dataframe_emails, 0.6, extra_tokens_to_remove = extra_tokens_to_remove)

# for i in numpy.sort(dataframe_emails['grouped_From'].unique()):
#     print(i)
#     print(dataframe_emails[dataframe_emails['grouped_From'] == i]['From'])
#     print('')




# """
# Filter out feedback emails and produce a balanced dataset based on Email type
# """
# # filter out feedback emails
# filtered_dataframe_emails = dataframe_emails[dataframe_emails['Email_type'] != 'Feedback']
# filtered_email_bodies = filtered_dataframe_emails.loc[:, filtered_dataframe_emails.columns != 'Email_type']
# filterer_email_types = filtered_dataframe_emails['Email_type']

# # define undersampling strategy and undersample to have a balanced dataset
# undersample_strategy = imblearn.under_sampling.RandomUnderSampler(sampling_strategy = 'majority', random_state = 0)
# undersampled_email_dataframe, undersampled_email_types = undersample_strategy.fit_resample(filtered_email_bodies, filterer_email_types)


# """
# Train supervised model to predict Email_type from tfidf embedding of email bodies
# """
# # create tfidf vectorizer for undersampled email bodies
# _, _, tfidf_body_vectorizer = build_tfidf_embedding_from_dataframe(undersampled_email_dataframe, 'Body', ngram_range = (1,2))

# # create test train split 
# [email_bodies_train, email_bodies_test, email_types_train, email_types_test] = sklearn.model_selection.train_test_split(undersampled_email_dataframe['Body'], undersampled_email_types, test_size = 0.2, random_state = 0, stratify =  undersampled_email_types)

# # create random forest classifier
# random_forest_classifier = sklearn.ensemble.RandomForestClassifier(n_estimators = 10)
# random_forest_classifier.fit(tfidf_body_vectorizer.transform(email_bodies_train), email_types_train)

# # evaluate confusion matrix performance of train dataset
# email_types_predicted = random_forest_classifier.predict(tfidf_body_vectorizer.transform(email_bodies_train))
# print('Confusion matrix train dataset')
# print(sklearn.metrics.confusion_matrix(email_types_train, email_types_predicted))
# print('Metrics report train dataset')
# print(sklearn.metrics.classification_report(email_types_train, email_types_predicted))






# # # perform nmf decomposition
# # NMF_model = sklearn.decomposition.NMF(n_components = 3, init = 'nndsvd', random_state = 0)
# # pattern_coefficients = NMF_model.fit_transform(tfidf_embedded_Bodies)
# # linguistic_patterns = pandas.DataFrame(NMF_model.components_, columns = embedding_Bodies_labels).transpose()

# # # identify topics of linguistic patterns
# # for i in linguistic_patterns.columns:
# #     temp = linguistic_patterns[i].sort_values(ascending = False)
# #     print('pattern ' + str(i))
# #     print(temp.head(n = 10))
# #     print('')

# # # cluster body contents with kmeans rather than NMF
# #     KMEans_model = sklearn.cluster.KMeans(n_clusters = 3, n_init= 100, random_state = 0)
# #     H = KMEans_model.fit_transform(tfidf_embedded_Bodies)
# #     topics = pandas.DataFrame(KMEans_model.cluster_centers_, columns = embedding_Bodies_labels).transpose()

# # for i in topics:
# #     temp = linguistic_patterns[i].sort_values(ascending = False)
# #     print('pattern ' + str(i))
# #     print(temp.head(n = 10))
# #     print('')






