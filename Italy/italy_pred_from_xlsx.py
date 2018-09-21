#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 10:30:55 2018

@author: niels-peter
"""

import numpy as np
from numpy import math
import pandas as pd
import pickle
from sklearn.externals import joblib
import re  
from italy_transformation import *


#clf_EW_italy = joblib.load('/home/niels-peter/Dokumenter/EW_DK_ITALY.pkl')
clf_EW_italy = joblib.load('EW_DK_ITALY_dummy.pkl')


#xl = pd.ExcelFile('/home/niels-peter/Dokumenter/ITALY_100_FINANCIAL_STATEMENT.xlsx')
xl = pd.ExcelFile('italy_database.xlsx')


output = []
input_d = [] 
for post in xl.sheet_names:
    df = xl.parse(post)
    columns =list(df)
    year = year_pre = 0
    for col in columns:
 #       print(col)
        try:
            col_year = int(col)
#            print(col_year)
            if col_year > year:
                year_pre = year
                year = col_year
            if year_pre < col_year < year:
                year_pre = col_year
        except:
            pass
    dict_data = {}
    for data in df[['ENGLISH', year_pre, year]].values:
        if data[0] ==  'Total active':
            dict_data['fsa:Assets'] = data[2]
            dict_data['fsa:Assets_prev'] = data[1]
        if data[0] ==  'B - Total fixed assets':
            dict_data['fsa:NoncurrentAssets'] = data[2]
            dict_data['fsa:NoncurrentAssets_prev'] = data[1]
        if data[0] ==  'B.I - Total intangible assets':
            dict_data['fsa:IntangibleAssets'] = data[2]
            dict_data['fsa:IntangibleAssets_prev'] = data[1]
        if data[0] ==  'B.II - Total tangible assets':
            dict_data['fsa:PropertyPlantAndEquipment'] = data[2]
            dict_data['fsa:PropertyPlantAndEquipment_prev'] = data[1]
        if data[0] ==  'B.III - Total financial fixed assets':
            dict_data['fsa:LongtermInvestmentsAndReceivables'] = data[2]
            dict_data['fsa:LongtermInvestmentsAndReceivables_prev'] = data[1]
        if data[0] ==  'C - Total working capital':
            dict_data['fsa:CurrentAssets'] = data[2]
            dict_data['fsa:CurrentAssets_prev'] = data[1]
        if data[0] ==  'C.I - Total inventories':
            dict_data['fsa:Inventories'] = data[2]
            dict_data['fsa:Inventories_prev'] = data[1]
        if data[0] ==  'C.II - Total credits':
            dict_data['fsa:ShorttermReceivables'] = data[2]
            dict_data['fsa:ShorttermReceivables_prev'] = data[1]
        if data[0] ==  'C.III - Total financial assets that do not constitute fixed assets':
            dict_data['fsa:ShorttermInvestments'] = data[2]
            dict_data['fsa:ShorttermInvestments_prev'] = data[1]
        if data[0] ==  'C.IV - Total liquid assets':
            dict_data['fsa:CashAndCashEquivalents'] = data[2]
            dict_data['fsa:CashAndCashEquivalents_prev'] = data[1]
        if data[0] ==  'A - Total equity':
            dict_data['fsa:Equity'] = data[2]
            dict_data['fsa:Equity_prev'] = data[1]
        if data[0] ==  'Difference between value and costs of production':
            dict_data['fsa:ProfitLossFromOrdinaryOperatingActivities'] = data[2]
            dict_data['fsa:ProfitLossFromOrdinaryOperatingActivities_prev'] = data[1]
        if data[0] ==  '23 - Profit (loss) for the year':
            dict_data['fsa:ProfitLoss'] = data[2]
            dict_data['fsa:ProfitLoss_prev'] = data[1]

    slettes = []
    for noegle in dict_data.keys():
        if math.isnan(dict_data[noegle]):
            slettes.append(noegle)
    for noegle in slettes:
        del dict_data[noegle]
    
    input_d.append(dict_data)
    
    transform_italy = italy_to_dict()
    transformed_data = transform_italy.transform([dict_data])
    output.append(transformed_data[0])

    print(clf_EW_italy.predict([dict_data]), clf_EW_italy.predict_proba([dict_data]), post, col_year)

#df_out = pd.DataFrame.from_dict(output)
#df_in = pd.DataFrame.from_dict(input_d)
#
#import matplotlib.pyplot as plt
##print(df_out.columns)
#
#col = []
#for p in list(df_out):
#    if 'ratio' in p:
#        col.append(p)
#
#fig, ax = plt.subplots(figsize =(25,20))
#df_out.hist(col, ax=ax, bins = 20)
#fig.savefig('italy.png')
    
