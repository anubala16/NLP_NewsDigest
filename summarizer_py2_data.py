import re
import nltk
import pandas as pd
abstract_bool = False
article_bool = False
abstracts = []
articles = []
sentences = {}
i = 0 # to keep track of article count
abstract_regex = re.compile(r'abstract\s*')
article_regex = re.compile(r'article\s*')

with open("data/50_train.bin", "rb") as textFile:
    line = textFile.readline().decode("utf-8")
    re.sub(r'[^\x00-\x7f]', r'', line)
    while line:
        print("Bin Line:", line)
        if article_bool is True or abstract_bool is True:
            if len(line) < 5: #see if line is blank (and should be ignored) or is actually important
                line = textFile.readline().decode("utf-8") # skip the blank, empty line and move to a line with some content
                re.sub(r'[^\x00-\x7f]', r'', line)
            # process the abstract/summary line
            line = line.strip()
            print("Trimmed Bin Line:", line)
            # break down into sentences

            # add to sentences {}

            if abstract_bool == True:
                #add to abstract list
                abstract_bool = False
                abstracts.append(line) #save the new abstract
            else: # article_bool is true
                # add to article list
                articles.append(line)
                article_bool = False #save the article
        abs_results = re.findall(abstract_regex, str(line)) #line has "abstract"
        if len(abs_results) >= 1:
            print("Abstract Line!")
            abstract_bool = True
        art_results = re.findall(article_regex, str(line))  # line has "article"
        if len(art_results) >= 1:
            print("Article Line")
            article_bool = True

        line = textFile.readline().decode("utf-8")
        re.sub(r'[^\x00-\x7f]', r'', line) #remove all hex special characters

def printList(list):
    for i in range(len(list)):
        print(i, ".", list[i])

print("Article sentences:")
printList(articles)

print("\nAbstracts:")
printList(abstracts)
