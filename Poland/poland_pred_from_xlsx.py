#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:24:10 2018

@author: niels-peter
"""

import numpy as np
import pandas as pd
from poland_transformation import *
import pickle
from sklearn.externals import joblib
from numpy import math
from poland_transformation import *

clf_EW_poland = joblib.load('/home/niels-peter/Dokumenter/EW_DK_POLAND.pkl')
df = pd.read_excel('/home/niels-peter/Dokumenter/Poland_Database.xlsx', header = [0, 1]) 
#clf_EW_poland = joblib.load('EW_DK_POLAND_dummy.pkl')
#df = pd.read_excel('poland_Database.xlsx', header = [0, 1])

output = []

datalist = df.values
for virksomhed in range(0, len(datalist), 1):
    dict_input_poland = {}
    for tekst, col in (('fsa:NoncurrentAssets_prev', 12),
                       ('fsa:NoncurrentAssets', 13),
                       ('fsa:IntangibleAssets_prev', 14),
                       ('fsa:IntangibleAssets', 15),
                       ('fsa:PropertyPlantAndEquipment_prev', 16),
                       ('fsa:PropertyPlantAndEquipment', 17),
                       ('fsa:LongtermInvestmentsAndReceivables_prev', 18),
                       ('fsa:LongtermInvestmentsAndReceivables', 19),
                       ('fsa:CurrentAssets_prev', 20),
                       ('fsa:CurrentAssets', 21),
                       ('fsa:Inventories_prev', 22),
                       ('fsa:Inventories', 23),
                       ('fsa:ShorttermReceivables_prev', 24),
                       ('fsa:ShorttermReceivables', 25),
                       ('fsa:ShorttermInvestments_prev', 26),
                       ('fsa:ShorttermInvestments', 27),
                       ('fsa:CashAndCashEquivalents_prev', 28),
                       ('fsa:CashAndCashEquivalents', 29),  
                       ('fsa:Prepayments_prev', 30),
                       ('fsa:Prepayments', 31),  
                       ('fsa:Equity_prev', 32),
                       ('fsa:Equity', 33),
                       ('fsa:Provisions_prev', 46),
                       ('fsa:Provisions', 47),
                       ('fsa:LiabilitiesOtherThanProvisions_prev', 48),
                       ('fsa:LiabilitiesOtherThanProvisions', 49),
                       ('fsa:ProfitLoss_prev', 92),
                       ('fsa:ProfitLoss', 93),
                       ):
        vaerdi = datalist[virksomhed][col]
        dict_input_poland[tekst] = vaerdi
    dict_input_poland['fsa:Assets'] = np.nan_to_num(dict_input_poland['fsa:NoncurrentAssets']) + np.nan_to_num(dict_input_poland['fsa:CurrentAssets']) + np.nan_to_num(dict_input_poland['fsa:Prepayments'])
    dict_input_poland['fsa:Assets_prev'] = np.nan_to_num(dict_input_poland['fsa:NoncurrentAssets_prev']) + np.nan_to_num(dict_input_poland['fsa:CurrentAssets_prev']) + np.nan_to_num(dict_input_poland['fsa:Prepayments_prev'])
    slettes = []
    for noegle in dict_input_poland.keys():
        if math.isnan(dict_input_poland[noegle]):
            slettes.append(noegle)
    for noegle in slettes:
        del dict_input_poland[noegle]
        
    transform_poland = Polish_to_dict()
    transformed_data = transform_poland.transform([dict_input_poland])
    output.append(transformed_data[0])
    
    #print(dict_input_poland)
    print(virksomhed, clf_EW_poland.predict_proba([dict_input_poland]))

#df_out = pd.DataFrame.from_dict(output)
#
#import matplotlib.pyplot as plt
##print(df_out.columns)
#
#df_out = df_out.drop(df_out.index[[53,86]])
#
#col = []
#for p in list(df_out):
#    if 'ratio' in p:
#        col.append(p)
#
#fig, ax = plt.subplots(figsize =(25,20))
#df_out.hist(col, ax=ax, bins = 20)
#fig.savefig('poland.png')
#
#hist_df = joblib.load('../hist_df.pkl')
#
#for column in col:
#    df_out['PL_' + column] = pd.cut(df_out[column], [-10000, -0.01, 0.2, 0.4, 0.6, 0.8, 1, 10000], labels=['<0', '0-0,2', '0,2-0,4', '0,4-0,6', '0,6-0,8', '0,8-1,0', '>1'])
#    #s2 = pd.Series(df_out['IT_' + column]).value_counts()
#    #s2 = s.value_counts()
#    hist_df = hist_df.append(pd.Series(df_out['PL_' + column]).value_counts())
#
#joblib.dump(hist_df, '../hist_df.pkl')
