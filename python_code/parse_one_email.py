#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:08:09 2020

@author: dabol99

This function parses the file_name and file_folder of an email and i returns
its contents. 
"""
# import relevant libraries
import email
from email import policy # odd, but I can't use email.policy without this
import re
import bs4

def parse_email(email_path):
    # open email from path in binary mode
    with open(email_path, 'rb') as imported_email:
        # instanciate a parser object for the email with default email policies
        email_parser = email.parser.BytesParser(policy = policy.default).parse(imported_email)
        
        # create dictionary with email metadata and its text
        # metadata
        email_dictionary = dict(zip(email_parser.keys(), email_parser.values()))
        # email bod - done by detecting simple message and by accessing the
        # relevant part in multipart messages. HTML makrdowns are stripped.
        if email_parser.is_multipart():
            # find body using MIME with html and plain text as preferred defaults.
            email_dictionary['Body'] = email_parser.get_body(preferencelist=('html', 'plain')).get_content()
        else:
            email_dictionary['Body'] = email_parser.get_body().get_content()
        
        # strip html markdowns
        email_dictionary['Body'] = bs4.BeautifulSoup(email_dictionary['Body'], "lxml").get_text() 
            
        # strip newlines, tabs and multiple whitespaces 
        email_dictionary['Body'] = re.sub('\n', ' ', email_dictionary['Body'])
        email_dictionary['Body'] = re.sub('\t', ' ', email_dictionary['Body'])
        email_dictionary['Body'] = re.sub('\s+', ' ', email_dictionary['Body'])
        email_dictionary['Body'] = re.sub('^ ', '', email_dictionary['Body'])
    return email_dictionary
