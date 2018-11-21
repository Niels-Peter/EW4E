#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:24:10 2018

@author: niels-peter
"""

import numpy as np
import pandas as pd
from spain_transformation import *
import pickle
from sklearn.externals import joblib
from numpy import math
import os
from spain_transformation import *

output = []

#clf_EW_spain = joblib.load('/home/niels-peter/Dokumenter/EW_DK_SPAIN.pkl')
clf_EW_spain = joblib.load('EW_DK_SPAIN_dummy.pkl')

#listen = os.listdir('/home/niels-peter/Dokumenter/100_spanske_Balance_Sheets')
listen = os.listdir('100_spanske_Balance_Sheets')



for forekomst in listen:
    #print(forekomst)
    #df = pd.read_excel('/home/niels-peter/Dokumenter/100_spanske_Balance_Sheets/' + forekomst)
    df = pd.read_excel('100_spanske_Balance_Sheets/' + forekomst)
    datalisten = df.values
    for i in range(0, len(datalisten)):
        if datalisten[i, 0] == 'Assets':
            datastart = i
            dataslut = len(datalisten[i])-1
            break
    dict_data = {}
    for i in range(datastart, len(datalisten)):
        if datalisten[i, 1] == 10000:
            dict_data['fsa:Assets'] = datalisten[i, dataslut]
            dict_data['fsa:Assets_prev'] = datalisten[i, dataslut-1]
        if datalisten[i, 1] == 11000:
            dict_data['fsa:NoncurrentAssets'] = datalisten[i, dataslut]
            dict_data['fsa:NoncurrentAssets_prev'] = datalisten[i, dataslut-1]
        if datalisten[i, 1] == 11100:
            dict_data['fsa:IntangibleAssets'] = datalisten[i, dataslut]
            dict_data['fsa:IntangibleAssets_prev'] = datalisten[i, dataslut-1]
        if datalisten[i, 1] == 12000:
            dict_data['fsa:CurrentAssets'] = datalisten[i, dataslut]
            dict_data['fsa:CurrentAssets_prev'] = datalisten[i, dataslut-1]
        if datalisten[i, 1] == 12200:
            dict_data['fsa:Inventories'] = datalisten[i, dataslut]
            dict_data['fsa:Inventories_prev'] = datalisten[i, dataslut-1]
        if datalisten[i, 1] == 12300:
            dict_data['fsa:ShorttermReceivables'] = datalisten[i, dataslut]
            dict_data['fsa:ShorttermReceivables_prev'] = datalisten[i, dataslut-1]
        if datalisten[i, 1] == 12700:
            dict_data['fsa:CashAndCashEquivalents'] = datalisten[i, dataslut]
            dict_data['fsa:CashAndCashEquivalents_prev'] = datalisten[i, dataslut-1]
        if datalisten[i, 1] == 20000:
            dict_data['fsa:Equity'] = datalisten[i, dataslut]
            dict_data['fsa:Equity_prev'] = datalisten[i, dataslut-1]
        if datalisten[i, 1] == 49500:
            dict_data['fsa:ProfitLoss'] = datalisten[i, dataslut]
            dict_data['fsa:ProfitLoss_prev'] = datalisten[i, dataslut-1]
        if datalisten[i, 1] == 49100:
            dict_data['fsa:ProfitLossFromOrdinaryOperatingActivities'] = datalisten[i, dataslut]
            dict_data['fsa:ProfitLossFromOrdinaryOperatingActivities_prev'] = datalisten[i, dataslut-1]
             
    slettes = []
    for noegle in dict_data.keys():
        if math.isnan(dict_data[noegle]):
            slettes.append(noegle)
    for noegle in slettes:
        del dict_data[noegle] 
        
    transform_spain = spain_to_dict()
    transformed_data = transform_spain.transform([dict_data])
    output.append(transformed_data[0])
            
    print(clf_EW_spain.predict([dict_data]), clf_EW_spain.predict_proba([dict_data]))
