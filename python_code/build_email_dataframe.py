#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:39:30 2020

@author: dabol99

This file contains the function that builds a dataframe from the email collection
stored in downloaded_emails_path
"""

# import libraries
import os 
import pandas
from parse_one_email import parse_email


def build_email_dataframe(downloaded_emails_path):
    # build list of email in downloaded_emails_path
    list_of_eml_files = [file for file in os.listdir(downloaded_emails_path) if file.endswith('.eml')]

    # parse all emails and store their bodies and metadata in one dataframe
    emails = [parse_email(downloaded_emails_path + '/' + email_name) for email_name in list_of_eml_files]
    dataframe_emails = pandas.DataFrame(emails).reindex(columns = ['Date', 'From', 'Subject', 'Body'])

    # get raw email and store it into a new field
    dataframe_emails['Sender_email'] = dataframe_emails['From'].str.extract(pat = '([\+\w\.-]+@[\w\.-]+)')

    # strip quoted text from emails and linkedin random text
    # based on "From:"
    dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('From:.+', '')
    # based on "Från: "
    dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('Från:.+', '')
    # based on linkedIn "Personal information Name "
    dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('Personal information Name .+', '')
    # based on linkedIn "View Message ©"
    dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('View Message ©.+', '')
    
 
    return dataframe_emails