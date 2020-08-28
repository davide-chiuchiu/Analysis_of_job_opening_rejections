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
import os 
import pandas
import numpy
import nltk
import seaborn
import re
import matplotlib.pyplot

# set current work directory to the one with this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import functions from auxiliary files
from parse_one_email import parse_email


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

# get company name.
# get raw email and remove it from the 'From' field
dataframe_emails['Sender_email'] = dataframe_emails['From'].str.extract(pat = '([\+\w\.-]+@[\w\.-]+)')
dataframe_emails['From'] = dataframe_emails.apply(lambda x: re.sub('<{0,1}' + x['Sender_email'].replace('+', '\+') + '>{0,1}', '', x['From']), axis = 1)


temp = dataframe_emails.reindex(columns = ['From', 'Sender_email'])

#dataframe_emails['Company_tentative_1'] = dataframe_emails['From'].str.extract(pat = '@([^>^"]+)>?')



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



