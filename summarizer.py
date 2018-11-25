from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from collections import OrderedDict
#from pyrouge import Rouge155
from rouge import Rouge
#from rougescore import rougescore
import glob
import os
import re

def printList(list):
    for i in range(len(list)):
        print(i, ".", list[i])

def printSentences(sents):
    for i in range(len(sents)):
        print("Sent Num:", sents[i][1][1], "Score:", sents[i][1][0], sents[i][0])

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

sent_path = 'data/sents'
summ_path = 'data/summs'

corpus=[]

print("Reading articles...")
for filename in sorted(glob.glob(os.path.join(sent_path, '*.sent'))):
    file=open(filename,"r",encoding="utf8")
    text=file.read()
    corpus.append(text)
    file.close()
print("Read", len(corpus), "articles.")

print("Reading summaries...")
given_summary=[]
for filename in sorted(glob.glob(os.path.join(summ_path, '*.summ'))):
    file=open(filename,"r",encoding="utf8")
    text=file.read()
    given_summary.append(text)
    file.close()
print("Read", len(given_summary), "summaries.")

vectorizer = TfidfVectorizer(stop_words='english')
# tokenize and build vocab

stop_words=set(stopwords.words('english'))

for art_idx in range(len(corpus)):
    print("Computing TFIDF scores for sentences in article", art_idx, "(ignoring stopwords)...")
    vectorizer.fit([corpus[art_idx]])
    # transform document into vector
    vector = vectorizer.transform([corpus[art_idx]]).toarray()
    print("Fit TFIDF model:", vector.shape)
    sentScore = dict()
    sent_num = 0
    for sent in sent_tokenize(corpus[art_idx]):
        for w in word_tokenize(sent):
            if w not in stop_words and hasNumbers(w) == False:
                index=vectorizer.vocabulary_.get(w)
                if(index!=None):
                     w_score=vector[0][index]
                     if sent[0:15] in sentScore:
                         sentScore[sent][0] += w_score
                     else:
                         sentScore[sent]=[w_score, sent_num]
        sent_num += 1
    sorted_sents = sorted(sentScore.items(), key=lambda w: (w[1][0]), reverse=True)  # sort by TF-IDF scores

    print("TF-IDF sentences:")
    printSentences(sorted_sents)
    sorted_sents = sorted_sents[:3]
    sorted_sents = sorted(sorted_sents, key=lambda x: x[1][1])
    print("\n\nFinal Summary:\n")
    printList(sorted_sents)
    print('-----------------------\n')

'''
reference=given_summary[2]
rouge = Rouge()
scores = rouge.get_scores(summary, reference)
print(scores)

r=rouge.rouge_n(summary,reference,1)
print(r)
'''
