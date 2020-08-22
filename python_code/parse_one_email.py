#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:08:09 2020

@author: dabol99

This function parses the file_name and file_folder of an email and i returns
its contents. 


SEE IF YOU CAN OUTPUT THE RELEVANT INFORMATIONS ALREADY AS A PANDAS DATAFRAME
"""
# import relevant libraries
import email
from email import policy # odd, but I can't use email.policy without this

def parse_email(email_path):
    # open email from path in binary mode
    with open(email_path, 'rb') as imported_email:
        # instanciate a parser object for the email with default email policies
        email_parser = email.parser.BytesParser(policy = policy.default).parse(imported_email)
        
        # create dictionary with email metadata and its text
        # metadata
        email_dictionary = dict(zip(email_parser.keys(), email_parser.values()))
        # email body
        email_dictionary['Body'] = email_parser.get_body().get_content()
    return email_dictionary
