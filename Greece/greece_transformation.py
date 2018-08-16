#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:24:10 2018

@author: niels-peter
"""

import numpy as np
from numpy import math
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class greece_to_dict(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        for dict in posts:            
            aktiver = ['fsa:NoncurrentAssets', 'fsa:IntangibleAssets','fsa:PropertyPlantAndEquipment', 
                       'fsa:LongtermInvestmentsAndReceivables', 'fsa:CurrentAssets', 'fsa:Inventories', 'fsa:ShorttermReceivables', 'fsa:ShorttermInvestments', 'fsa:CashAndCashEquivalents',
                       'fsa:LiabilitiesOtherThanProvisions', 'fsa:LongtermLiabilitiesOtherThanProvisions', 'fsa:ShorttermLiabilitiesOtherThanProvisions',
                       'fsa:Provisions']
            passiver = ['fsa:Equity', 'fsa:LiabilitiesOtherThanProvisions', 'fsa:LongtermLiabilitiesOtherThanProvisions', 'fsa:ShorttermLiabilitiesOtherThanProvisions',
                        'fsa:Provisions']            
            if ('fsa:Assets' in dict) or ('fsa:ProfitLoss' in dict) or ('fsa:Equity' in dict):
                for felt in ('fsa:Assets',
                             'fsa:NoncurrentAssets',  'fsa:IntangibleAssets',  'fsa:PropertyPlantAndEquipment',  'fsa:LongtermInvestmentsAndReceivables',
                             'fsa:CurrentAssets',  'fsa:Inventories',  'fsa:ShorttermReceivables', 'fsa:ShorttermInvestments', 'fsa:CashAndCashEquivalents',  
                             'fsa:Equity', 'fsa:LiabilitiesOtherThanProvisions', 'fsa:LongtermLiabilitiesOtherThanProvisions', 'fsa:ShorttermLiabilitiesOtherThanProvisions', 'fsa:Provisions',
                             'fsa:GrossResult', 'fsa:GrossProfitLoss', 'fsa:ProfitLossFromOrdinaryOperatingActivities', 'fsa:ProfitLoss'):                    
                    if felt in dict:
                        try:
                            dict[felt] = float(dict.get(felt, 0))
                        except:
                            pass
                    if felt in aktiver:
                        try:
                            dict[felt + '_ratio'] = dict[felt] / dict['fsa:Assets']
                            dict[felt + '_ratio'].replace({np.inf: 9, -np.inf: -9})
                        except:
                            pass
                    if felt == 'fsa:Equity':
                            try:
                                dict['Soliditetsgrad'] = dict[felt] / dict['fsa:Assets']
                                dict['Soliditetsgrad'].replace({np.inf: 9, -np.inf: -9})
                            except:
                                pass
                    if felt in passiver:
                        try:
                            dict[felt + '_ratio'] = dict[felt] / (dict['fsa:Assets']- dict['fsa:Equity'])
                            dict[felt + '_ratio'].replace({np.inf: 9, -np.inf: -9})
                        except:
                            pass
                try:
                    dict['egenkap_forrentning'] = dict['fsa:ProfitLoss']/float(dict['fsa:Equity'])
                    dict['egenkap_forrentning'].replace({np.inf: 9, -np.inf: -9})
                    if dict['egenkap_forrentning'] > 9:
                        dict['egenkap_forrentning'] = 9
                    if dict['egenkap_forrentning'] < -9:
                        dict['egenkap_forrentning'] = -9
                except:
                    pass
                if ('fsa:GrossResult' in dict) and ('fsa:GrossProfitLoss' in dict):
                    dict['GrossProfit'] = max(dict.get('fsa:GrossResult', None), dict.get('fsa:GrossProfitLoss', None))
                elif ('fsa:GrossResult' in dict) and ('fsa:GrossProfitLoss' not in dict):
                    dict['GrossProfit'] = dict.get('fsa:GrossResult', None)
                elif ('fsa:GrossResult' not in dict) and ('fsa:GrossProfitLoss' in dict):
                    dict['GrossProfit'] = dict.get('fsa:GrossProfitLoss', None)
                try:
                    dict['GrossProfit_to_OrdinaryOperatingActivities_ratio'] = (dict['GrossProfit'] - dict['fsa:ProfitLossFromOrdinaryOperatingActivities'])/dict['GrossProfit']
                    dict['GrossProfit_to_OrdinaryOperatingActivities_ratio'].replace({np.inf: 9, -np.inf: -9})
                    if dict['GrossProfit_to_OrdinaryOperatingActivities_ratio'] > 9:
                        dict['GrossProfit_to_OrdinaryOperatingActivities_ratio'] = 9
                    if dict['GrossProfit_to_OrdinaryOperatingActivities_ratio'] < -9:
                        dict['GrossProfit_to_OrdinaryOperatingActivities_ratio'] = -9
                        
                except:
                    pass                    
                try:
                    dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio'] = (dict['fsa:ProfitLossFromOrdinaryOperatingActivities'] - dict['fsa:ProfitLoss'])/dict['GrossProfit']
                    dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio'].replace({np.inf: 9, -np.inf: -9})
                    if dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio'] > 9:
                        dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio'] = 9
                    if dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio'] < -9:
                        dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio'] = -9
                except:
                    pass 
                        
                if ('fsa:Assets_prev' in dict) or ('fsa:ProfitLoss_prev' in dict) or ('fsa:Equity_prev' in dict):
                    for felt in ('fsa:Assets',
                                 'fsa:NoncurrentAssets',  'fsa:IntangibleAssets',  'fsa:PropertyPlantAndEquipment',  'fsa:LongtermInvestmentsAndReceivables',
                                 'fsa:CurrentAssets',  'fsa:Inventories',  'fsa:ShorttermReceivables', 'fsa:ShorttermInvestments', 'fsa:CashAndCashEquivalents',  
                                 'fsa:Equity', 'fsa:LiabilitiesOtherThanProvisions', 'fsa:LongtermLiabilitiesOtherThanProvisions', 'fsa:ShorttermLiabilitiesOtherThanProvisions', 'fsa:Provisions',
                                 'fsa:GrossResult', 'fsa:GrossProfitLoss', 'fsa:ProfitLossFromOrdinaryOperatingActivities', 'fsa:ProfitLoss'):                    

                        if felt + '_prev' in dict:
                            try:
                                dict[felt + '_prev'] = float(dict.get(felt + '_prev', 0)) 
                            except:
                                pass
                            try:
                                dict[felt + '_delta'] = (dict[felt] - dict[felt + '_prev'])/dict[felt + '_prev']
                                dict[felt + '_delta'].replace({np.inf: 9, -np.inf: -9})
                            except:
                                pass
                        if felt == 'fsa:Equity':
                            try:
                                dict['Soliditetsgrad_prev'] = dict['fsa:Equity_prev'] / dict['fsa:Assets_prev']
                                dict['Soliditetsgrad_prev'].replace({np.inf: 9, -np.inf: -9})
                            except:
                                pass
                    try:
                        dict['egenkap_forrentning_prev'] = dict['fsa:ProfitLoss_prev'] /float(dict['fsa:Equity_prev'])
                        dict['egenkap_forrentning_prev'].replace({np.inf: 9, -np.inf: -9})
                        if dict['egenkap_forrentning_prev'] > 9:
                            dict['egenkap_forrentning_prev'] = 9
                        if dict['egenkap_forrentning_prev'] < -9:
                            dict['egenkap_forrentning_prev'] = -9
                    except:
                        pass
                    if ('fsa:GrossResult_prev' in dict) and ('fsa:GrossProfitLoss_prev' in dict):
                        dict['GrossProfit_prev'] = max(dict.get('fsa:GrossResult_prev', None), dict.get('fsa:GrossProfitLoss_prev', None))
                    elif ('fsa:GrossResult_prev' in dict) and ('fsa:GrossProfitLoss_prev' not in dict):
                        dict['GrossProfit_prev'] = dict.get('fsa:GrossResult_prev', None)
                    elif ('fsa:GrossResult_prev' not in dict) and ('fsa:GrossProfitLoss_prev' in dict):
                        dict['GrossProfit_prev'] = dict.get('fsa:GrossProfitLoss_prev', None)
                    try:
                        dict['GrossProfit_to_OrdinaryOperatingActivities_ratio_prev'] = (dict['GrossProfit_prev'] - dict['fsa:ProfitLossFromOrdinaryOperatingActivities_prev'])/dict['GrossProfit_prev']
                        dict['GrossProfit_to_OrdinaryOperatingActivities_ratio_prev'].replace({np.inf: 9, -np.inf: -9})
                        if dict['GrossProfit_to_OrdinaryOperatingActivities_ratio_prev'] > 9:
                            dict['GrossProfit_to_OrdinaryOperatingActivities_ratio_prev'] = 9
                        if dict['GrossProfit_to_OrdinaryOperatingActivities_ratio_prev'] < -9:
                            dict['GrossProfit_to_OrdinaryOperatingActivities_ratio_prev'] = -9
                    except:
                        pass
                    try:
                        dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio_prev'] = (dict['fsa:ProfitLossFromOrdinaryOperatingActivities_prev'] - dict['fsa:ProfitLoss_prev'])/dict['GrossProfit_prev']
                        dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio_prev'].replace({np.inf: 9, -np.inf: -9})
                        if dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio_prev'] > 9:
                            dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio_prev'] = 9
                        if dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio_prev'] < -9:
                            dict['OrdinaryOperatingActivities_to_ProfitLoss_ratio_prev'] = -9
                    except:
                        pass

        return [{'Assets_delta': dict.get('fsa:Assets_delta', 0),
                 'NoncurrentAssets_delta': dict.get('fsa:NoncurrentAssets_delta', 0),
                 'NoncurrentAssets_ratio': dict.get('fsa:NoncurrentAssets_ratio', 0),
                 'IntangibleAssets_delta': dict.get('fsa:IntangibleAssets_delta', 0),  
                 'IntangibleAssets_ratio': dict.get('fsa:IntangibleAssets_ratio', 0),  
                 'PropertyPlantAndEquipment_delta': dict.get('fsa:PropertyPlantAndEquipment_delta', 0),
                 'PropertyPlantAndEquipment_ratio': dict.get('fsa:PropertyPlantAndEquipment_ratio', 0),
                 'LongtermInvestmentsAndReceivables_delta': dict.get('fsa:LongtermInvestmentsAndReceivables_delta', 0),
                 'LongtermInvestmentsAndReceivables_ratio': dict.get('fsa:LongtermInvestmentsAndReceivables_ratio', 0),
                 'CurrentAssets_delta': dict.get('fsa:CurrentAssets_delta', 0),
                 'CurrentAssets_ratio': dict.get('fsa:CurrentAssets_ratio', 0),
                 'Inventories_delta': dict.get('fsa:Inventories_delta', 0),
                 'Inventories_ratio': dict.get('fsa:Inventories_ratio', 0),
                 'ShorttermReceivables_delta': dict.get('fsa:ShorttermReceivables_delta', 0),
                 'ShorttermReceivables_ratio': dict.get('fsa:ShorttermReceivables_ratio', 0),
                 'ShorttermInvestments_delta': dict.get('fsa:ShorttermInvestments_delta', 0),
                 'ShorttermInvestments_ratio': dict.get('fsa:ShorttermInvestments_ratio', 0),
                 'CashAndCashEquivalents_delta': dict.get('fsa:CashAndCashEquivalents_delta', 0),
                 'CashAndCashEquivalents_ratio': dict.get('fsa:CashAndCashEquivalents_ratio', 0),
                 'Equity_delta': dict.get('fsa:Equity_delta', 0),
                 'Solvency ratio': dict.get('Soliditetsgrad', 0),
                 'Solvency ratio_prev': dict.get('Soliditetsgrad_prev', 0),                 
                 'LiabilitiesOtherThanProvisions_delta': dict.get('fsa:LiabilitiesOtherThanProvisions_delta', 0),
                 'LiabilitiesOtherThanProvisions_ratio': dict.get('fsa:LiabilitiesOtherThanProvisions_ratio', 0),
                 'Provisions_delta': dict.get('fsa:Provisions_delta', 0),
                 'Provisions_ratio': dict.get('fsa:Provisions_ratio', 0),
                 'Return on Equity': dict.get('egenkap_forrentning', 0),
                 'Return on Equity_prev': dict.get('egenkap_forrentning_prev', 0),                     
                 'GrossProfit_to_OrdinaryOperatingActivities_ratio': dict.get('GrossProfit_to_OrdinaryOperatingActivities_ratio', 0),
                 'GrossProfit_to_OrdinaryOperatingActivities_ratio_prev': dict.get('GrossProfit_to_OrdinaryOperatingActivities_ratio_prev', 0),
                 'OrdinaryOperatingActivities_to_ProfitLoss_ratio': dict.get('OrdinaryOperatingActivities_to_ProfitLoss_ratio', 0),
                 'OrdinaryOperatingActivities_to_ProfitLoss_ratio_prev': dict.get('OrdinaryOperatingActivities_to_ProfitLoss_ratio_prev', 0),
                 } for dict in posts]
    