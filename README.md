# Blood Bank ChatBot

### Description

A chatbot implementation from scratch to help resolve human queries about the blood camp .

### Undersatnding the database - 
The donation camps are spread across 5 districts of the state Andhra Pradesh in India ; namely Kurnool , Guntur , Kadapa , Chitoor and Anantapur (district table) . Each district has a hospital where the camp is active (hospital table). The hospital table also gives information about the number of units of each blood group left at the hospitals . 

- Only district and hospital tables have been used in the Data Service Layer for quering the database through the chat implementation . Work has to be extended upto querying the entire database and updating the same .

>Note - Refer the ER Diagram also

### Requirements
Python modules  :
- nltk
- numpy
- tensorflow
- tflearn
- pandas
- json
- string
- random

>Note - tensorflow-gpu is recommended while training the classifier

### Features
1. Human style conversations
2. Intent classification
3. Multiple intent handling
4. Handling context
4. Multiple Entity extraction
5. Form action while handling intents
6. Database interaction through Data Service Layer for resolving database specific queries.


### Further Work
1) Custom NER to be used for multiple entity extraction . (A case specific approach has been used in this implementation ).
2) Handling context using RNNS and LSTMs .
3) UI to be built.


### References 
- https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077


> Feel free to submit your chatflows through issues to help improve the chatbot implementation .



