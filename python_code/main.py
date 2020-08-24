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

# set current work directory to the one with this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import functions from auxiliary files
from parse_one_email import parse_email


# get list of files within the downloaded_emails folder
downloaded_emails_path = os.path.dirname(os.getcwd()) + '/downloaded_emails'
list_of_eml_files = [file for file in os.listdir(downloaded_emails_path) if file.endswith('.eml')]

# parse all emails and store their bodies and metadata in one dataframe
emails = [parse_email(downloaded_emails_path + '/' + email_name) for email_name in list_of_eml_files]
dataframe_emails = pandas.DataFrame(emails).loc[:, ['Body', 'Date', 'From', 'Sender', 'Subject', 'Type']]