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


#clf_EW_poland = joblib.load('/home/niels-peter/Dokumenter/EW_DK_POLAND.pkl')
#df = pd.read_excel('/home/niels-peter/Dokumenter/Poland_Database.xlsx', header = [0, 1]) 
clf_EW_poland = joblib.load('EW_DK_POLAND_dummy.pkl')
df = pd.read_excel('Poland_Database.xlsx', header = [0, 1])


datalist = df.as_matrix()
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
        if str(vaerdi) == 'nan':
            dict_input_poland[tekst] = 0
        else:
            dict_input_poland[tekst] = vaerdi
    dict_input_poland['fsa:Assets'] = dict_input_poland['fsa:NoncurrentAssets'] + dict_input_poland['fsa:CurrentAssets'] + dict_input_poland['fsa:Prepayments']
    dict_input_poland['fsa:Assets_prev'] = dict_input_poland['fsa:NoncurrentAssets_prev'] + dict_input_poland['fsa:CurrentAssets_prev'] + dict_input_poland['fsa:Prepayments_prev']
    print(dict_input_poland)
    print(virksomhed, clf_EW_poland.predict([dict_input_poland]))
