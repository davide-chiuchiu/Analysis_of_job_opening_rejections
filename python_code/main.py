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


# set current work directory to the one with this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import functions from auxiliary files
from parse_one_email import parse_email
from text_utilities import cluster_emails_by_From

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



