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
import nltk
import os 
import sklearn
import sklearn.ensemble
import imblearn




# set current work directory to the one with this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import functions from auxiliary files
from build_email_dataframe import build_email_dataframe
from text_utilities import compute_word_frequencies
from compute_rejection_days import compute_rejection_days
from plots import plot_email_type_distribution, plot_email_lengths_distribution
from plots import save_keyword_wordclouds_of_email_types, plot_boxplot_of_days_to_reject_distribution


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
# build buzzwords from automated routines and educated guesses in extra_buzzword_guess
extra_buzzwords_guess = ['car', 'hir', 'jobvit', 'lev', 'system', 'successfact', 'team']
days_to_reject = compute_rejection_days(dataframe_emails, extra_buzzwords_guess = extra_buzzwords_guess)
plot_boxplot_of_days_to_reject_distribution(days_to_reject)



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