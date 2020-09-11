#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:39:30 2020

@author: dabol99

This file contains the functions that builds a dataframe from the email collection
stored in downloaded_emails_path
"""

# import libraries
import os 
import pandas
from parse_one_email import parse_email
from text_utilities import preprocess_corpus


# set current work directory to the one with this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


"""
This function takes the emails in subfolder from downloaded emails_path, and 
it stores their relevant content in partiak_dataframe_emails. The function
also labels the emails based on their subffolder origin
"""
def build_email_dataframe_from_subfolder(downloaded_emails_path, subfolder):
    # build subfolder directory
    email_subfolder_path = downloaded_emails_path + '/' + subfolder

    # build list of email in email_subfolder_path
    list_of_eml_files = [file for file in os.listdir(email_subfolder_path) if file.endswith('.eml')]

    # parse all emails and store their bodies and metadata in one dataframe
    emails = [parse_email(email_subfolder_path + '/' + email_name) for email_name in list_of_eml_files]
    partial_dataframe_emails = pandas.DataFrame(emails).reindex(columns = ['Date', 'From', 'Subject', 'Body'])
    
    # add labeling to emails
    partial_dataframe_emails['Email_type'] = subfolder 
    
    return partial_dataframe_emails




"""
This function builds a dataframe from the email collection in downloaded_emails_path
and it labels them based on the subfolders in downloaded_emails_path  where 
they are stored. The function also implents a few filtering options to strip 
junk text from the email bodies, and to tokenize&stem the email bodies.
"""
def build_email_dataframe(downloaded_emails_path):
    # build list of subdirectories within downloaded_emails_path
    list_of_subdirectories = [subdirectory for subdirectory in os.listdir(downloaded_emails_path) if not subdirectory.startswith('.')]

    # create list of dataframes with emails from each subdirectory
    partial_email_dataframes = [build_email_dataframe_from_subfolder(downloaded_emails_path, subdirectory) for subdirectory in list_of_subdirectories]

    # merge partial email dataframes in one single dataframe
    dataframe_emails = pandas.concat(partial_email_dataframes).reset_index().reindex(columns = ['Date', 'From', 'Subject', 'Body', 'Email_type'])

    # strip quoted text from emails and linkedin random text
    # based on "From:"
    dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('From:.+', '')
    # based on "Från: "
    dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('Från:.+', '')
    # based on linkedIn "Personal information Name "
    dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('Personal information Name .+', '')
    # based on linkedIn "View Message ©"
    dataframe_emails['Body'] = dataframe_emails['Body'].str.replace('View Message ©.+', '')
    
    # Tokenize and stem email bodies with preprocess_corpus
    dataframe_emails['Stemmed Body'] = dataframe_emails.apply(lambda x: preprocess_corpus(x['Body']) , axis = 1)

    # get raw email and store it into a new field
    dataframe_emails['Sender_email'] = dataframe_emails['From'].str.extract(pat = '([\+\w\.-]+@[\w\.-]+)')
    
 
    return dataframe_emails