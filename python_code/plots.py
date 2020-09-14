#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:12:13 2020

@author: dabol99

This file contains the code for all the plots I perform in the main.py function
"""

import matplotlib
import os
import seaborn
import wordcloud

"""
This function takes the dataframe_email and it returns the histogram plot of 
email type distribution
"""
def plot_email_type_distribution(dataframe_emails):  
    # create figure and axis
    email_type_distribution_figure, email_type_distribution_axis = matplotlib.pyplot.subplots()
    # plot histogram with seaborn
    seaborn.countplot(x = 'Email_type', data = dataframe_emails, ax = email_type_distribution_axis)
    # save figure in the Latex_summary_report file
    plot_destination_file = os.path.dirname(os.getcwd()) + '/Latex_summary_report/email_type_distribution.eps'
    email_type_distribution_figure.savefig(plot_destination_file, format='eps')
    
    return email_type_distribution_figure

"""
This functino usese the word_frequencies_dataframe containing the word frequencis 
for each email type, to build and save the wordclouds of the keywords in each
email type.  
"""
def save_keyword_wordclouds_of_email_types(word_frequencies_dataframe):
    # initialize wordcloud object
    word_cloud_object = wordcloud.WordCloud(background_color = 'white', random_state = 0, max_words = 30)
    # initialize figure and figure axis
    wordcloud_figure, wordcloud_axis = matplotlib.pyplot.subplots()
    for email_type in word_frequencies_dataframe.index:
        # build word cloud for email_type
        wordcloud_instance = word_cloud_object.generate_from_frequencies(word_frequencies_dataframe[email_type])
        # plot and save wordcloud
        wordcloud_axis.imshow(wordcloud_instance)
        wordcloud_figure.suptitle('Word cloud for ' + email_type  + ' candidacy emails')
        matplotlib.pyplot.axis('off')
        plot_destination_file = os.path.dirname(os.getcwd()) + '/Latex_summary_report/wordcloud_' + email_type + '.png'
        wordcloud_figure.savefig(plot_destination_file, format = 'png', dpi = 300)
    
    return 