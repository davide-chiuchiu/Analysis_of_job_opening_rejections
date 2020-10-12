#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:04:51 2020

@author: dabol99

This file contains one major function, compute_rejection_time_for_candidacy
which computes the time it took to beeing rejected for each job application.
To this end, the function first uses hierarchical clustering to identify emails 
from the same company, then it computes the days passes between the candidacy submission
and the candidacy rejection.
"""
from cluster_emails_by_From import cluster_emails_by_From, define_buzzwords_for_From_field


def compute_rejection_days(dataframe_emails, extra_buzzwords_guess = []):
    buzzwords = define_buzzwords_for_From_field(dataframe_emails['Processed sender and subject'], extra_buzzwords = extra_buzzwords_guess)
    # cluster emails together based on the sender and subject information
    dataframe_emails['Indexed sender'] = cluster_emails_by_From(dataframe_emails, 0.5, method = "average", extra_tokens_to_remove = buzzwords)

    # save cluster information to check if the current parameters are good enough
    automated_email_groups_destination = '../automated_email_grouping.csv'
    dataframe_emails[['Indexed sender', 'From']].sort_values(by = ['Indexed sender']).reset_index().to_csv(automated_email_groups_destination)

    # find number of emails and the date of the first email
    # within emails from the same cluster/sender and same email type
    aggregated_date_email_number_info = dataframe_emails\
        .groupby(['Indexed sender', 'Email_type'])\
        .agg(first_email_date = ('Date', min), email_counts = ('Date', len))\
        .reset_index()

    # find clusters/senders that has at least one 'Received' email and one 'Rejected' email
    list_of_sender_clusters_with_reception_email = aggregated_date_email_number_info[aggregated_date_email_number_info['Email_type'] == 'Received']['Indexed sender'].unique()
    list_of_sender_clusters_with_rejection_email = aggregated_date_email_number_info[aggregated_date_email_number_info['Email_type'] == 'Rejected']['Indexed sender'].unique()
    list_of_sender_cluster_with_reception_and_rejection_email = list(set(list_of_sender_clusters_with_reception_email) & set(list_of_sender_clusters_with_rejection_email))

    # select aggregated informations only for groups in 
    #list_of_sender_cluster_with_reception_and_rejection_email and reshape the dataframe
    filtered_aggregated_date_email_number_info = aggregated_date_email_number_info[aggregated_date_email_number_info['Indexed sender'].isin(list_of_sender_cluster_with_reception_and_rejection_email)] \
        .drop(columns = 'email_counts') \
        .pivot(index = 'Indexed sender', columns = 'Email_type', values = 'first_email_date')

    # compute days time it takes to reject and filter out anomalies where 
    # the day count is negative
    days_to_reject = filtered_aggregated_date_email_number_info.apply(lambda x: (x['Rejected'] - x['Received']).days, axis = 1 ).rename("Days before rejection")
    days_to_reject = days_to_reject[days_to_reject > 0]

    return days_to_reject