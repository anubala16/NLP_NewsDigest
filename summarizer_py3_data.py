from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from collections import OrderedDict
from rouge import Rouge

import glob
import os

sents_path = 'data/sents/'
summs_path = 'data/summs/'

corpus=[]
for filename in sorted(glob.glob(os.path.join(sents_path, '*.sent'))):
    file=open(filename,"r",encoding="utf8")
    text=file.read()
    corpus.append(text)

corpus_summary=[]
for filename in sorted(glob.glob(os.path.join(summs_path, '*.summ'))):
    file=open(filename,"r",encoding="utf8")
    text=file.read()
    corpus_summary.append(text)

vectorizer = TfidfVectorizer(stop_words='english')

# tokenize and build vocab
vectorizer.fit(corpus)

# transform document into vector
vector = vectorizer.transform([corpus[2]]).toarray()


# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights1 = sorted(vector, key=lambda w: w[1], reverse=True)

stop_words=set(stopwords.words('english'))
sentScore=dict()    

for sent in sent_tokenize(corpus[2]):
    for w in word_tokenize(sent):
        if w not in stop_words:
            index=vectorizer.vocabulary_.get(w)
            if(index!=None):
                 w_score=vector[0][index]
                 if sent[0:15] in sentScore:
                     sentScore[sent]+=w_score
                 else:
                     sentScore[sent]=w_score
        

sorted_dict = OrderedDict(sorted(sentScore.items(), key=lambda x: x[1],reverse=True))

print("\n\nTF-IDF Summary for article 2:\n------------------------------\n")
i=0
summ=""
for k, v in sorted_dict.items():
    if(i>3):
        break  
    print("%s\n" % (k))
    summ+=k
    i+=1

print(summ)

summary=summ

reference=corpus_summary[2]
rouge = Rouge()
scores = rouge.get_scores(summary, reference)
