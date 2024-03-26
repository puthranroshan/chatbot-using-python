import nltk
import tensorflow as tf
import numpy as np
import random 
import json
import pickle
import nltk
nltk.download('punkt')
import nltk
nltk.download('wordnet')
#Pickling in Python is the process of converting a Python object (such as a list, dictionary, etc.) into a byte stream that can be saved on disk or sent over a network This is also known as serialization
from nltk.stem import WordNetLemmatizer
#WordNet database to perform lemmatization. Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item
# For example, the words "rocks", "rocking", and "rocked" can be lemmatized to the base form "rock".
lemmatizer = WordNetLemmatizer()
intents=json.load(open('intents.json'))

words =[]
classes =[]
documents=[]
ignoreletters=['?','!','.',','] # ignore these letters if they come in data

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordlist=nltk.word_tokenize(pattern) #it will create a list of tokens 
        words.extend(wordlist)
        # the words.append(iterable ) will take an iterable as an argument
        #the itrable can be a list or any tupple which has multiple values
        documents.append((wordlist,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words= [lemmatizer.lemmatize(word) for word in words if word not in ignoreletters]

words=sorted(set(classes))#sorting the words based on the classes

classes=sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
# dump() function is part of the pickle module.It is used to serialize an object
# open('words.pkl', 'wb'):
# The open() function is used to open a file.
# The first argument ('words.pkl') specifies the filename.
# The second argument ('wb') specifies the mode in which the file should be opened:
# 'w': Write mode
# 'b': Binary mode
# So 'wb' means “write binary”,

training = []

outputempty =[0]*len(classes)

for document in documents:
    bag=[]
    wordpatterns =document[0]
    wordpatterns = [lemmatizer.lemmatize(word.lower()) for word in wordpatterns]
    for word in words: bag.append(1) if word in wordpatterns else bag.append(0)

    outputrow=list(outputempty)
    outputrow[classes.index(document[1])] = 1
    training.append(bag + outputrow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))#drooping the 50 % data
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))#gradient desent

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')

        


