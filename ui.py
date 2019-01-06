#The Web UI part

from flask import Flask,render_template ,redirect, url_for , request
# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import sqlite3 as sql
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import nltk
from nltk.corpus import stopwords

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

import pandas as pd
import string

import re
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
import nltk.collocations
import nltk.corpus
import json


word_list = word_tokenize(open('words.txt').read())
punctuations = ['^', ')', '/', '#', '{', '=','-', '~', '|', '`', '&', '$', '_', ',', '\\', '?', "'", '[', '(', ']', '*', '"', ':', '}', '%', '<', '.', '>', '!', '@', '+', ';']

def processLanguage(text):
    new_sent =[]
    for t in text:
        if t not in punctuations:
            new_sent.append(t)
        elif t in punctuations:
            new_sent.append(' ')
            new_sent.append(t)
            new_sent.append(' ')
    new_sent = ''.join(new_sent)
    sentences = sent_tokenize(new_sent)
    
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
        return [i.lower() for i in proper_nouns]
    except Exception as e:
        print(str(e))

with open('new_intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

#print (len(documents), "documents")
#print (len(classes), "classes", classes)
#print (len(words), "unique stemmed words", words)

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

#print(training[:,1])

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])


tf.reset_default_graph()
# Build neural network
table_net = tflearn.input_data(shape=[None, len(train_x[0])])
table_net = tflearn.fully_connected(table_net, 8)
table_net = tflearn.fully_connected(table_net, 8)
table_net = tflearn.fully_connected(table_net, len(train_y[0]), activation='softmax')
table_net = tflearn.regression(table_net)

# Define model and setup tensorboard
model = tflearn.DNN(table_net, tensorboard_dir='table_tflearn_logs')

# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('new_intents.json') as json_data:
    intents = json.load(json_data)



# load our saved model
model.load('./model.table_tflearn')

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    #results = [[i,r] for i,r in enumerate(results)]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list



def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    results = [i[0] for i in results]
    #results = multi_intent(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and context[userID] in i['context_filter']):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return i['function']
                    else:
                        return "context_not_set_handler"
            results.pop(0)
def multi_intent(query):
    res = [i[0] for i in classify(query)]
    #print(res)
    if(len(res)>1):
        #print(res)
        print("Multiple intents found")
        return res
    else:
        sentences = sent_tokenize(query)
        #print(sentences)
        if(len(sentences)>1):
            intents = []
            for i in sentences:
                #print(classify(i))
                sub_intents = classify(i)
                for j in sub_intents:
                    intents.append(j[0])
            #print(intents)
            intents = list(set(intents))
            if('time' in intents):
                intents.remove('time')
                intents.append('time')
            elif('greeting' in intents):
                intents.remove('greeting')
                intents = ['greeting']+intents
            return intents
        else:
            words = word_tokenize(query)
            sentences = [' '.join(words[:int(len(words)/2)]),' '.join(words[int(len(words)/2):])]
            #print(sentences)
            intents = []
            for i in sentences:
                #print(classify(i))
                sub_intents = classify(i)
                for j in sub_intents:
                    if(j[1]>0.5):
                        intents.append(j[0])
            #print(intents)
            intents = list(set(intents))
            if('time' in intents):
                intents.remove('time')
                intents.append('time')
            elif('greeting' in intents):
                intents.remove('greeting')
                intents = ['greeting']+intents
            return intents

#connecting to database
conn=sql.connect("blood.db")

def capture_context(user_chat_history):
    for i in user_chat_history:
        result = classify(i)[0][0]
        print(i,end = "-->")
        print(result)
        #print()



tables = ['district','bill_payment','blood_recepient','donor','hospital']

def district_handler(query):
    result = classify(query)[0][0]
    b_grp = ['a+','a-','b+','b-','ab+','ab-','o+','o-']
    w = [i.lower() for i in word_tokenize(query)]
    blood_group = [i for i in w if i in b_grp]
    #print(blood_group)
    if(len(blood_group)>0):
        query = query.lower()
        query = query.split(' ')
        for i in query:
            if blood_group[0] in i:
                query.remove(i)

        query = ' '.join(query)
        blood_group = blood_group[0]
    #print(query)
    else:
        blood_group = None
    details = dict()
    details['table'] = "district"
    details['named_entities'] = processLanguage(query)
    if(len(details['named_entities'])==0):
        command = "select dname from district;"
        res = list(conn.execute(command))
        #[('kurnool',), ('chitoor',), ('anantapur',), ('kadapa',), ('guntur',)]
        l = ["We operate in \n","We have our camps in the following districts\n","You can visit the closest among the following districts\n"]
        s = random.choice(l)
        for i in res:
            s = s + i[0]+"\n"
        return s
    l = []
    #implement a classifier to indentify if a query is of boolean , select or aggregate type
    #assuming the type of query is a boolean query
    for i in details['named_entities']:
        #print(i)
        #have to identify the column/columns in target
        #print(i)
        command = "select * from "+ details['table'] +" where dname= '"+i+"';"
        #print(command)
        res = list(conn.execute(command))
        #print(res)
        conn.commit()
        if(len(res)>0):
            l.append("Yes")
        else:
            l.append("No")
    if(len(l)>0):
        return l[0]
    else:
        return "Yet to resolve this query"


'''
def get_ent(query):
    nlp2 = spacy.load("/home/pruthvi/Desktop/Thought_Clan/blood_bank_querybot/my_ner")
    s = query.lower()
    stop_words = list(set(stopwords.words('english')))
    doc = nlp2(s)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    for i in entities:
        if(i[0] in stop_words):
            entities.remove(i)
    
    ents = [i[0] for i in entities]
    named_ents = processLanguage(query)
    
    place = []
    for i in named_ents:
        for j in ents:
            if(i in j):
                place.append(i)
            
    print(entities,place)
'''


def get_ent(query,donor=False,recepient=False):
    b_grp = ['a+','a-','b+','b-','ab+','ab-','o+','o-']
    w = [i.lower() for i in word_tokenize(query)]
    blood_group = [i for i in w if i in b_grp]
    #print(blood_group)
    if(len(blood_group)>0):
        query = query.lower()
        query = query.split(' ')
        for i in query:
            if blood_group[0] in i:
                query.remove(i)

        query = ' '.join(query)
        blood_group = blood_group[0]
    #print(query)
    else:
        blood_group = None
    named_ents = processLanguage(query)
    #print(named_ents)
    if(len(named_ents)>0):
        location = named_ents[0]
    else:
        location = None
    units = re.findall(r'\d+', query)
    units = [int(i) for i in units]
    if(len(units)>1):
        unit = min(units)
        age = max(units)
    elif(len(units)==1):
        if(units[0]<10):
            unit = units[0]
            age = None
        else:
            unit = None
            age = units[0]
    else:
        unit = None
        age = None
    #if(units!=None):
    #    units = units.group()
    #print(units)
    
    if(donor==True):
        ent_dict = {'age':age,'blood_group':blood_group,'location':location}
        return ent_dict
    elif(recepient==True):
        ent_dict = {'age':age,'blood_group':blood_group,'location':location,'units_required':unit}
        return ent_dict    

    

def context_not_set_handler(query):
    return "Will have to set context before asking this ..."
    
def greeting_handler(query):
    l = ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help?"]
    return random.choice(l)
    
def goodbye_handler(query):
    l = ["See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon."]
    return random.choice(l)
    
def thanks_handler(query):
    l = ["Happy to help!", "Any time!", "My pleasure"]
    return random.choice(l)

def time_handler(query):
    l = ["The camps in the hospitals open at 9AM and close by 11PM ."]
    return random.choice(l)

def want2donate_handler(query):
    l = ["Glad to hear that.","Good to know.","Pleased to hear that"]
    print("bot reply - "+random.choice(l)+"\nI might need a few more details")
    questions = dict()
    donor_details = dict()
    questions['age'] = ['How old are you ?','May I know your age ?','What is your age ?']
    questions['blood_group'] = ['What is your blood group ?',"May I know your blood group ?"]
    questions['location'] = ['Name the city/town you live in .','Which city do you stay ?',"Where do you live ?","Please tell me your place of stay."]
    
    donor_details['age'] =[]
    donor_details['blood_group'] = []
    donor_details['location'] =[]
    
    ent_dict = get_ent(query,donor=True)
    for i in ent_dict.keys():
        if(ent_dict[i]!=None):
            donor_details[i].append(ent_dict[i])
    
    b_grp = ['a+','a-','b+','b-','ab+','ab-','o+','o-']
    
    while(len(donor_details['age'])==0 or len(donor_details['blood_group'])==0 or len(donor_details['location'])==0):
        a = random.choice(list(donor_details.keys()))
        if(len(donor_details[a])==0):
            print(random.choice(questions[a]))
            if(a=='age'):
                try:
                    donor_details['age'].append(int(input()))
                except:
                    print("Please enter valid age")
                    while(True):
                        try:
                            donor_details['age'].append(int(input()))
                            break
                        except:
                            print("Please enter valid age")
                            
            elif(a=='blood_group'):
                while(True):
                    w = [i.lower() for i in word_tokenize(input())]
                    bl_grp = [i for i in w if i in b_grp]
                    if(len(bl_grp)>0):
                        donor_details['blood_group'].append(bl_grp[0])
                        break
                    else:
                        print("Please enter valid blood group")
            elif(a=='location'):
                while(True):
                    w = processLanguage(input())
                    if(len(w)>0):
                        donor_details['location'].append(w[0])
                        break
                    else:
                        print("Please enter a valid location")
                    
        else:
            continue
    
    print("Here are your details:")
    for i in list(donor_details.keys()):
        print( i ,":", donor_details[i][0])
    
    if(donor_details['age'][0]<=50):
        print("We would be drawing a maximum of 2 units of blood (350ml) from you .")
    else:
        print("We would be drawing a unit of blood (350 ml) from you .")
    districts = ["Kurnool" , "Guntur" , "Kadapa" , "Chitoor","Anantapur"]
    dis = random.choice(districts)
    print(dis+" should be the closest to your place.")
    
    print("Please visit any government hospital in "+dis+" district to donate blood")
    print("We appreciate your willingness . Thank you")
    return ''

def want2receive_handler(query):
    l = ["Okay","I see","Sure","Oh okay","Oh"]
    print("bot reply - "+random.choice(l)+"\nI might need a few more details")
    questions = dict()
    recepient_details = dict()
    questions['age'] = ['How old are you ?','May I know your age ?','What is your age ?']
    questions['blood_group'] = ['What is your blood group ?',"May I know your blood group ?"]
    questions['location'] = ['Name the city/town you live in .','Which city do you stay ?',"Where do you live ?","Please tell me your place of stay."]
    questions['units_required'] = ['How many units of blood do you want ?','How many units of blood are required ?','How many units do you want ?',"What's the quantity in need (in terms of units)?"]
    
    recepient_details['age'] =[]
    recepient_details['blood_group'] = []
    recepient_details['location'] =[]
    recepient_details['units_required'] = []
    
    ent_dict = get_ent(query,recepient=True)
    for i in ent_dict.keys():
        if(ent_dict[i]!=None):
            recepient_details[i].append(ent_dict[i])
    
    b_grp = ['a+','a-','b+','b-','ab+','ab-','o+','o-']
    
    while(len(recepient_details['age'])==0 or len(recepient_details['blood_group'])==0 or len(recepient_details['location'])==0 or len(recepient_details['units_required'])==0):
        a = random.choice(list(recepient_details.keys()))
        if(len(recepient_details[a])==0):
            print(random.choice(questions[a]))
            if(a=='age'):
                try:
                    recepient_details['age'].append(int(input()))
                except:
                    print("Please enter valid age")
                    while(True):
                        try:
                            recepient_details['age'].append(int(input()))
                            break
                        except:
                            print("Please enter valid age")
            elif(a=='blood_group'):
                while(True):
                    w = [i.lower() for i in word_tokenize(input())]
                    bl_grp = [i for i in w if i in b_grp]
                    if(len(bl_grp)>0):
                        recepient_details['blood_group'].append(bl_grp[0])
                        break
                    else:
                        print("Please enter valid blood group")
                    
            elif(a=='location'):
                while(True):
                    w = processLanguage(input())
                    if(len(w)>0):
                        recepient_details['location'].append(w[0])
                        break
                    else:
                        print("Please enter a valid location")
                    
            elif(a=='units_required'):
                try:
                    recepient_details['units_required'].append(int(input()))
                except:
                    print("Please enter a number")
                    while(True):
                        try:
                            recepient_details['units_required'].append(int(input()))
                            break
                        except:
                            print("Please enter a number")
        else:
            continue
    
    print("Here are your details:")
    for i in list(recepient_details.keys()):
        print( i ,":", recepient_details[i][0])
    
    
    districts = ["Kurnool" , "Guntur" , "Kadapa" , "Chitoor","Anantapur"]
    #dis = random.choice(districts)
    #print(dis+" should be the closest to your place.")
    
    for dis in districts:
        command = "select dis_id from district where dname='"+dis.lower()+"';"
        did = list(conn.execute(command))[0][0][1]
        conn.commit()
        
        command = "select hname,hb_qty,h_id from hospital where dis_id = '"+did+"' and hb_grp='"+recepient_details['blood_group'][0].upper()+"';"
        b_qty = list(conn.execute(command))[0][1]
        hname =list(conn.execute(command))[0][0]
        h_id =list(conn.execute(command))[0][2]
        if(int(b_qty)>=int(recepient_details['units_required'][0])):
            print("Requirement found at hospital ",hname,"in",dis,"district ")
            print("You might be charged ",100*recepient_details['units_required'][0])
            print("Please visit the hospital ")
            print("Hope we have helped . Thank you")
            
            #command = "update hospital set hb_qty="+str(b_qty-int(recepient_details['units_required'][0]))+" where h_id ='"+str(h_id)+"' and hb_grp ='"+recepient_details['blood_group'][0].upper()+"';"                           
            
            
            return ''
    
    print("Couldn't find requirement . Extremely sorry")
    return ''

def no_reply_handler(query):
    return '...'

def chat():
    user_chat_history = []
    bot_chat_history = []
    print("Hello there !")
    query = ""
    while(classify(query)[0][0]!='goodbye'):
        query = input()
        user_chat_history.append(query)
        #s = response(query,show_details=True)
        s = response(query)
        #print(s)
        reply = globals()[s](query)
        '''
        if(s in tables):
            #reply = db_action(query,s)
            reply = globals()[s+"_handler"](query,s)
        else:
            reply = s
        '''
        bot_chat_history.append(reply)
        print("bot reply - ",reply)
        print()
    #capture_context(user_chat_history)
    #return (user_chat_history,bot_chat_history)



app = Flask(__name__)
@app.route('/home')
def home():
    
    
    return render_template("homepage.html")

if __name__ == '__main__':
    app.run(debug=True)


