# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:32:53 2018

@author: Tejaswini
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from collections import OrderedDict

file=open("data/article1.txt","r")
text1=file.read()
'''
file=open("article2.txt","r")
text2=file.read()

file=open("article3.txt","r")
text3=file.read()
'''
corpus=[text1,text1]

vectorizer = TfidfVectorizer(stop_words='english')
# tokenize and build vocab
vectorizer.fit(corpus)
#print(vectorizer.vocabulary)
# transform document into vector
vector1 = vectorizer.transform([text1]).toarray()
print("Vector 1 shape:", vector1.shape)
#vector2 = vectorizer.transform([text2]).toarray()

#vector3=  vectorizer.transform([text3]).toarray()

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights1 = sorted(vector1, key=lambda w: w[1], reverse=True)

stop_words=set(stopwords.words('english'))
sentScore=dict()    

for sent in sent_tokenize(text1):
    for w in word_tokenize(sent):
        if w not in stop_words:
            index=vectorizer.vocabulary_.get(w)
            if(index!=None):
                 w_score=vector1[0][index]
                 print(index)
                 if sent[0:15] in sentScore:
                     sentScore[sent]+=w_score
                 else:
                     sentScore[sent]=w_score
        

sorted_dict = OrderedDict(sorted(sentScore.items(), key=lambda x: x[1],reverse=True))

print("\n\nSummary:\n")
i=0
for k, v in sorted_dict.items():
    if(i>5):
        break  
    print("%s\n" % (k))
    i+=1

