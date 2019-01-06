#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 12:54:13 2019

@author: pruthvi
"""
import pandas as pd

sents = ["I'd be needing - units of blood",'I want - units of blood','- units are required','- packects are required','I need - packets','I want - packets','Do you have - units of blood ?','Do you have - packets of blood ?',"I'd be needing - packets of blood .","I'd be needing - bottles of blood",'I want - bottles of blood','- bottles are required','Do you have - bottles of blood ?']

unit_train_sentences = []

for sent in sents:
    for j in range(1,101):
        s = sent
        a = s.index('-')
        b = a+len(str(j))
        k = s.replace('-',str(j))
        unit_train_sentences.append('("'+k+'",{"entities":[('+str(a)+','+str(b)+',"UNIT"'+')]})')

print(','.join(unit_train_sentences))


'''
df = pd.read_csv('city.csv')
nums = list(df['numbers'])
nums = list(set(nums))
print(nums.pop(0))


for sent in sents:
    for j in nums:
        s = sent
        a = s.index('-')
        b = a+len(j)
        k = s.replace('-',j)
        unit_train_sentences.append('("'+k+'",{"entities":[('+str(a)+','+str(b)+',"UNIT"'+')]})')
'''

