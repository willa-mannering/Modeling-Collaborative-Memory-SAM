#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:18:41 2020

Group Recall
@author: willamannering
"""

from SAM_Group_Categorized import SAM_Group_Categorized
from SAM_Nominal_Categorized import SAM_Nominal_Categorized
import numpy as np
from scipy import stats

def update_group_response(group, group_response):
    for g in group:
        g.group_response = group_response
        
def get_fastest_response(r, accum_time): #helper function for group_context_recall
    
    if accum_time == [0]*len(accum_time):
    
        fastest_response = min(r)[1] #fastest response of any of the group members
        response_time = min(r)[0] #timestamp for fastest response
        
    else:
        for i in range(len(accum_time)):
            r[i][0] = r[i][0] + max(accum_time[i])
                
        fastest_response = min(r)[1]
        response_time = min(r)[0]
        
    return fastest_response, response_time, r

def group_context_recall(group, accum_time): 
    #all models start off doing context recall at the same time. Whichever finishes first "wins" and that #response is added

    r = []
    for g in group:
        r.append(g.context_recall())
        
    if min(r)[1] == -1: #if fastest model's response = -1, then all models failed to retrieve
        fastest_response = -1

    else:
        fastest_response, response_time, r = get_fastest_response(r, accum_time)
        
        for i in range(len(r)):#for each of the group members
            if r[i][0] == response_time: #if current model produced fastest response, don't need to do anything for model 
                
                group[i].update_assoc(fastest_response)#update associations for fastest response

            elif r[i][1] == -1: #if current model didn't produce a response, nothing happens because it's already reached kmax
                continue
            else:
                total_times = accum_time[i] + r[i][3]
                count_fails = len([f for f in total_times if f > response_time])#count how many retrieval failures happened after the fastest response
                group[i].K = group[i].K - count_fails
                group[i].update_assoc(fastest_response) #update associations involved with fastest response
    
    return fastest_response, accum_time, r
    
def group_wordcue_recall(cue, group, accum_time):
    #all models start off doing wordcue recall at the same time. Whichever finishes first "wins" and that response is added to the group_response vector
    r = []
    for g in group:
        
        if g.K >= g.Kmax:
            r.append(g.extra_wordcue_recall(cue)) #do wordcue recall even if past kmax
            
        else:
            r.append(g.wordcue_recall(cue))
        
    fastest_response = min(r)[1] #fastest response of any of the group members
    response_time = min(r)[0] #timestamp for fastest response

    if fastest_response == -1: #if no models produce a response record time accumulation because all models have reached lmax
        accum_time = []
        for i in range(len(group)): 
            accum_time.append(r[i][3])
        
    else: #if any model successfully produced a response
        for i in range(len(r)):
            if r[i][0] == response_time: #don't need to do anything for model that produced the min response
                group[i].update_assoc(fastest_response, cue)
                
            elif r[i][1] == -1: #if a model didn't produce a response
                count_fails = len([f for f in r[i][3] if f > response_time])#count how many retrieval failures happened after the fastest response
                group[i].K = group[i].K - count_fails #get rid of Ks that happened after fastest response
                group[i].update_assoc(fastest_response, cue) #update association with fastest response
                 
            else:
                count_fails = len([f for f in r[i][3] if f > response_time])#count how many retrieval failures happened after the fastest response
                group[i].K = group[i].K - count_fails
                group[i].update_assoc(fastest_response, cue) #update associations involved with fastest response
                
    return fastest_response, accum_time

def individual_recall(num_runs, list_length, category_size):

    len_response = []

    for i in range(num_runs):
        sam = SAM_Nominal_Categorized(list_length, category_size)
        len_response.append(len(sam.free_recall()))

    return len_response

def nominal_recall(group):
    nominal_response = []

    for g in group:
        r = g.free_recall()
        nominal_response.extend(r)
        
    return set(nominal_response)

def group_recall(group):
    
    group_response = []
    accum_recall_times = [[0]]*len(group) #keep track of each models accumulative time until one produces a response
    
    while( any([g for g in group if g.K < g.Kmax])): #while at least one model hasn't reached Kmax yet, keep recalling

        #begin with context_recall
        g1 = group_context_recall(group, accum_recall_times)
        
        if (g1[0] == -1): #if no model is able to retrieve a memory, that means all models reached kmax, so recall ends
            break
        
        else: #otherwise, add "winning" response to group response 
            accum_recall_times = [0]*len(group) #reset accumulated recall times to 0
            current_response = g1[0]
            
            group_response.append(current_response) #add current response to total group response
            update_group_response(group, group_response) #update internal group response tracker for individual models
        
        while(current_response != -1): #if cue has been recalled via context, use that cue to continue recall
        #once a cue is recalled, use that cue for recall
            g2 = group_wordcue_recall(current_response, group, accum_recall_times)
            if (g2[0] == -1):
                accum_recall_times = g2[1] #if all models reach lmax and produce no response, keep track of accum times for context recall
                break
            
            else:
                r = g2[0]

                group_response.append(r) #add new response to group_response

                update_group_response(group, group_response) #update internal group response tracker for individual models
                current_response = r
                     
    return group_response

def run_group_recall(numruns, list_length, category_size, group_size):
#do group recall X amount of times
    
    len_collab = [] #length of total collaborative recall response per run
    len_nom = [] #length of total nominal response per run

    for i in range(numruns):
        nominal_group = []
        collab_group = []
        individual = []

        for i in range(group_size):
            nominal_group.append(SAM_Nominal_Categorized(list_length, category_size))
            collab_group.append(SAM_Group_Categorized(list_length, category_size))

        #perform individual recall
        individual.append(individual_recall(numruns, list_length, category_size))

        #perform nominal recall
        nominal_response = nominal_recall(nominal_group)
        len_nom.append(len(nominal_response))

        #perfrom collaborative recall
        group_response = group_recall(collab_group)
        len_collab.append(len(group_response))

        
    print('individual: ', np.mean(individual)/list_length, np.std(individual)/list_length,'\nnominal: ', np.mean(len_nom)/list_length, np.std(len_nom)/list_length, 
    '\ncollaborative: ', np.mean(len_collab)/list_length, np.std(len_collab)/list_length)
    print('Collaborative statistically different from nominal? ', stats.ttest_ind(len_nom, len_collab))


def main():
    print(run_group_recall(10, 90, 15, 3))

if __name__ == "__main__":
    main()  

       















    
        
        
        
