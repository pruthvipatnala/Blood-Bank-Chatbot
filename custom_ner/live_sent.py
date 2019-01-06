"""
I am from ----

I live in ----

I'm from ----

I come from ----

I stay in ----
"""



import pandas as pd

df = pd.read_csv('city.csv')

cities = list(df['Cities'])
cities = [i.lower() for i in cities]

sents = ['I am from -','I live in -',"- is my hometown",'I come from -','I stay in -',"My residence is in -",'- is where my residence is .','-']

location_train_sentences = []

'''
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]
'''


for i in cities:
    for j in sents:
        s = j
        a = s.index('-')
        b = a+len(i)
        k = s.replace('-',i)        
        location_train_sentences.append('("'+k+'",{"entities":[('+str(a)+','+str(b)+',"LOC"'+')]})')


print(','.join(location_train_sentences))












