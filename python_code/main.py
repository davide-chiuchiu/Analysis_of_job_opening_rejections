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

# compute the distribution of days it takes for a candidacy to be rejected
extra_buzzwords_guess = ['car', 'hir', 'jobvit', 'lev', 'system', 'successfact', 'team']
days_to_reject = compute_rejection_days(dataframe_emails, extra_buzzwords = extra_buzzwords_guess)
plot_boxplot_of_days_to_reject_distribution(days_to_reject)

# identify companies that reject in less than 10 days. 
indexed_fast_rejecters = days_to_reject[days_to_reject <= days_to_reject.quantile(0.5)].index.to_list()
fast_rejecters = dataframe_emails[dataframe_emails['Indexed sender'].isin(indexed_fast_rejecters)]['Processed sender and subject']

indexed_moderate_rejecters = days_to_reject[(days_to_reject > days_to_reject.quantile(0.5)) & (days_to_reject <= days_to_reject.quantile(0.75))].index.to_list()
moderate_rejecters = dataframe_emails[dataframe_emails['Indexed sender'].isin(indexed_moderate_rejecters)]['Processed sender and subject']

indexed_slow_rejecters = days_to_reject[days_to_reject > days_to_reject.quantile(0.75)].index.to_list()
slow_rejecters = dataframe_emails[dataframe_emails['Indexed sender'].isin(indexed_slow_rejecters)]['Processed sender and subject']