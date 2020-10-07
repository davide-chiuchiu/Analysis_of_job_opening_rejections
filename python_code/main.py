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
import numpy
import nltk
import os 
import sklearn
import sklearn.ensemble
import imblearn




# set current work directory to the one with this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import functions from auxiliary files
from build_email_dataframe import build_email_dataframe

from cluster_emails_by_From import cluster_emails_by_From, define_buzzwords_for_From_field
from text_utilities import compute_word_frequencies
from plots import plot_email_type_distribution, plot_email_lengths_distribution
from plots import save_keyword_wordclouds_of_email_types
# build dataframe of emails from database of stored emails. The building procedure 
# remove numbers, and it stems the email bodies after tokenization
downloaded_emails_path = os.path.dirname(os.getcwd()) + '/downloaded_emails'
dataframe_emails = build_email_dataframe(downloaded_emails_path)

"""
Prelimianry analysis
"""
# inspect imbalance between email types
Email_type_distribution = dataframe_emails['Email_type'].value_counts(normalize = False)
print(plot_email_type_distribution(dataframe_emails))


# count number of sentences in each email and show the distribution based on email types and plot it 
dataframe_emails['Sentences count'] = dataframe_emails['Body'].apply(lambda x: len(nltk.tokenize.sent_tokenize(x)))
plot_email_lengths_distribution(dataframe_emails)


# build word frequencies in each email type and the associated word clouds
grouped_dataframe_emails = dataframe_emails.groupby('Email_type')
word_frequencies_dataframe = grouped_dataframe_emails.apply(lambda x: compute_word_frequencies(x['Stemmed Body']))
save_keyword_wordclouds_of_email_types(word_frequencies_dataframe)

"""
Identify distribution of rejection time
"""
# build buzzwords from automated routines and some educated guess
extra_buzzwords_guess = ['car', 'hir', 'jobvit', 'lev', 'system', 'successfact', 'team']
buzzwords = define_buzzwords_for_From_field(dataframe_emails['Processed sender and subject'], extra_buzzwords = extra_buzzwords_guess)
# cluster emails together based on the sender and subject information
dataframe_emails['Indexed sender'] = cluster_emails_by_From(dataframe_emails, 0.5, method = "average", extra_tokens_to_remove = buzzwords)

# clustering is far from perfect with these parameter choices for the cluster
# and the height to cut the tree, but decent enough.
for i in numpy.sort(dataframe_emails['Indexed sender'].unique()):
    print(i)
    print(dataframe_emails[dataframe_emails['Indexed sender'] == i]['From'])
    print('')

# find number of emails and the date of the first email
# within emails from the same cluster and same email type
aggregated_date_email_number_info = dataframe_emails\
    .groupby(['Indexed sender', 'Email_type'])\
    .agg(first_email_date = ('Date', min), email_counts = ('Date', len))\
    .reset_index()

# find indexed senders that hace 
list_of_sender_clusters_with_reception_email = aggregated_date_email_number_info[aggregated_date_email_number_info['Email_type'] == 'Received']['Indexed sender'].unique()
list_of_sender_clusters_with_rejection_email = aggregated_date_email_number_info[aggregated_date_email_number_info['Email_type'] == 'Rejected']['Indexed sender'].unique()
list_of_sender_cluster_with_reception_and_rejection_email = list(set(list_of_sender_clusters_with_reception_email) & set(list_of_sender_clusters_with_rejection_email))

# select aggregated informations only for groups in list_of_sender_cluster_with_reception_and_rejection_email
# and reshape the dataframe
filtered_aggregated_date_email_number_info = aggregated_date_email_number_info[aggregated_date_email_number_info['Indexed sender'].isin(list_of_sender_cluster_with_reception_and_rejection_email)] \
    .drop(columns = 'email_counts') \
    .pivot(index = 'Indexed sender', columns = 'Email_type', values = 'first_email_date')




# find groups with rejection emails.
indexed_senders_with_rejection_emails = numpy.sort(dataframe_emails[dataframe_emails['Email_type'] == 'Rejected']['Indexed sender'].unique())
dataframe_emails_with_rejections = dataframe_emails[dataframe_emails['Indexed sender'].isin(indexed_senders_with_rejection_emails)]
date_of_first_email_by_sender_and_type = dataframe_emails_with_rejections\
    .groupby(['Indexed sender', 'Email_type'])\
    .apply(lambda x: min(x['Date']))\
    .reset_index(name = 'Date')\
    .pivot(index = 'Indexed sender', columns = 'Email_type', values = 'Date')



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