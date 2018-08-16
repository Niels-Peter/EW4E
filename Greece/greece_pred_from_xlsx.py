#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:24:10 2018

@author: niels-peter
"""

import numpy as np
import pandas as pd
from greece_transformation import *
import pickle
from sklearn.externals import joblib
from numpy import math

#clf_EW_GREECE = joblib.load('/home/niels-peter/Dokumenter/EW_DK_GREECE.pkl')
clf_EW_GREECE = joblib.load('EW_DK_GREECE_dummy.pkl')

df = pd.read_excel('greece_database.xlsx', header = [0, 1])
#df = pd.read_excel('/home/niels-peter/Dokumenter/Greece_DBA_Project deliverable_20180312.xlsx', header = [0, 1]) 
headers = list(df)
lines = df.index.get_values()

datalist = df.values

#for post in range(0, 4, 2):
for post in range(0, len(headers), 2):
    #print(headers[post+1][0], headers[post+1][1], headers[post][1])
    dict_data = {}
    dict_data['fsa:Assets'] = datalist[:,post+1][0]
    dict_data['fsa:Assets_prev'] = datalist[:,post][0]
    dict_data['fsa:NoncurrentAssets'] = datalist[:,post+1][3]
    dict_data['fsa:NoncurrentAssets_prev'] = datalist[:,post][3]
    dict_data['fsa:IntangibleAssets'] = datalist[:,post+1][4]
    dict_data['fsa:IntangibleAssets_prev'] = datalist[:,post][4]
    dict_data['fsa:PropertyPlantAndEquipment'] = datalist[:,post+1][5]
    dict_data['fsa:PropertyPlantAndEquipment_prev'] = datalist[:,post][5]
    dict_data['fsa:LongtermInvestmentsAndReceivables'] = datalist[:,post+1][6]
    dict_data['fsa:LongtermInvestmentsAndReceivables_prev'] = datalist[:,post][6]
    dict_data['fsa:CurrentAssets'] = datalist[:,post+1][7]
    dict_data['fsa:CurrentAssets_prev'] = datalist[:,post][7]
    dict_data['fsa:Inventories'] = datalist[:,post+1][8]
    dict_data['fsa:Inventories_prev'] = datalist[:,post][8]
    dict_data['fsa:ShorttermReceivables'] = datalist[:,post+1][9]
    dict_data['fsa:ShorttermReceivables_prev'] = datalist[:,post][9]
    dict_data['fsa:ShorttermInvestments'] = datalist[:,post+1][10]
    dict_data['fsa:ShorttermInvestments_prev'] = datalist[:,post][10]
    dict_data['fsa:CashAndCashEquivalents'] = datalist[:,post+1][11]
    dict_data['fsa:CashAndCashEquivalents_prev'] = datalist[:,post][11]
    dict_data['fsa:Equity'] = datalist[:,post+1][14]
    dict_data['fsa:Equity_prev'] = datalist[:,post][14]    
    dict_data['fsa:Equity'] = datalist[:,post+1][14]
    dict_data['fsa:Equity_prev'] = datalist[:,post][14]
    dict_data['fsa:Provisions'] = datalist[:,post+1][21]
    dict_data['fsa:Provisions_prev'] = datalist[:,post][21]
    dict_data['fsa:LiabilitiesOtherThanProvisions'] = datalist[:,post+1][22]
    dict_data['fsa:LiabilitiesOtherThanProvisions_prev'] = datalist[:,post][22]
    dict_data['fsa:GrossProfitLoss'] = datalist[:,post+1][26]
    dict_data['fsa:GrossProfitLoss_prev'] = datalist[:,post][26]
    dict_data['fsa:ProfitLossFromOrdinaryOperatingActivities'] = datalist[:,post+1][26]-np.nan_to_num(datalist[:,post+1][27])-np.nan_to_num(datalist[:,post+1][28])
    dict_data['fsa:ProfitLossFromOrdinaryOperatingActivities_prev'] = datalist[:,post][26]-np.nan_to_num(datalist[:,post][27])-np.nan_to_num(datalist[:,post][28])
    dict_data['fsa:ProfitLoss'] = datalist[:,post+1][38]
    dict_data['fsa:ProfitLoss_prev'] = datalist[:,post][38]
    slettes = []
    for noegle in dict_data.keys():
        if math.isnan(dict_data[noegle]):
            slettes.append(noegle)
    for noegle in slettes:
        del dict_data[noegle]

    print(headers[post+1][0], clf_EW_GREECE.predict_proba([dict_data]))
