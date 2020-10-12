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
This function plots the distribution of email length based on email types from 
the emails in dataframe_email
"""
def plot_email_lengths_distribution(dataframe_emails):
    # create figure and axis
    email_length_distribution_figure, email_length_distribution_axis = matplotlib.pyplot.subplots()
    # plot boxplots with seaborn
    seaborn.boxplot(x = 'Sentences count', y = 'Email_type', data = dataframe_emails)
    # save figure in the Latex_summary_report folder
    plot_destination_file = os.path.dirname(os.getcwd()) + '/Latex_summary_report/message_length_distribution.eps'
    email_length_distribution_figure.savefig(plot_destination_file, format='eps')

    return(email_length_distribution_figure)




"""
This function usese the word_frequencies_dataframe containing the word frequencis 
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





"""
This function uses the information from the days_to_reject series to plot the
boxplot of the days it takes to have a candidacy rejected
"""
def plot_boxplot_of_days_to_reject_distribution(days_to_reject):
    # create figure and axis
    rejection_time_distribution_figure, rejedction_time_distribution_axis = matplotlib.pyplot.subplots()
    # plot boxplot
    seaborn.boxplot(x = days_to_reject)
    rejedction_time_distribution_axis.set_xticks([i for i in range(0, 90, 10)])
    # save figure in the Latex_summary_report file
    plot_destination_file = os.path.dirname(os.getcwd()) + '/Latex_summary_report/days_to_reject_distribution.eps'
    rejection_time_distribution_figure.savefig(plot_destination_file, format='eps')
