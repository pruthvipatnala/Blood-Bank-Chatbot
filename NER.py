#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:24:13 2017

@author: pruthvi
"""
import re
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
import nltk.collocations
import nltk.corpus
from geopy.geocoders import Nominatim
geolocator = Nominatim()


word_list = word_tokenize(open('words.txt').read())
punctuations = ['^', ')', '/', '#', '{', '=','-', '~', '|', '`', '&', '$', '_', ',', '\\', '?', "'", '[', '(', ']', '*', '"', ':', '}', '%', '<', '.', '>', '!', '@', '+', ';']

def processLanguage(sentences):
    try:
        proper_nouns =[]
        for item in sentences:
            l = word_tokenize(item.lower())
            s = re.compile(r'\.')
            p = s.sub(r' . ',' '.join(l))
            words = word_tokenize(p)
            for i in range(len(words)):
                if words[i] not in word_list:
                    words[i] = words[i].capitalize()
                    #print(words[i].capitalize())
            item = ' '.join(words)
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
            #print tagged
            namedEnt = nltk.ne_chunk(tagged)
            #namedEnt.draw()
            #print(namedEnt)
            for ent in namedEnt:    
                if type(ent)==nltk.tree.Tree:
                    if type(ent[0]) == tuple:
                        #print(ent[0][0])
                        proper_nouns.append(ent[0][0])
                #print(ent[0])
        return proper_nouns
    except Exception as e:
        print(str(e))

#print(punctuations)

#text = "Two police personnel were suspended after a video of them, brutally beating up two children on suspicion of theft in Bulandshahr, went viral."
#text = "JP Nagar . A youngster killed two boys by pushing them into Polavaram Right Main Canal to continue his alleged illicit relationship with their mother at Dippakayalapadu village of Koyyalagudem mandal in West Godavari. According to Jangared-dygudem DSP Ch. Murali Krishna, one Prasanth, 10, was studying Class V and his younger brother Vicky, 8, was studying Class III at a local school at Dippakalayapadu village."
text = input("Enter text :  ")
punct_list =[]
new_sent =[]
for t in text:
    if t not in punctuations:
        new_sent.append(t)
    elif t in punctuations:
        new_sent.append(' ')
        new_sent.append(t)
        new_sent.append(' ')
new_sent = ''.join(new_sent)

#print(new_sent)
print(processLanguage(sent_tokenize(new_sent)))

print(new_sent)

print(sent_tokenize(new_sent))

'''
locations=[]
for item in processLanguage(sent_tokenize(new_sent)):
    if geolocator.geocode(item)!=None:
        locations.append(item)
print(locations)
'''







