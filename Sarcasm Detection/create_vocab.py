# Create bag of words dictionary for unigram features
import codecs
import re
sarcasm_file=codecs.open('../../data/sarcasm_tweets.txt', encoding='utf-8')
nonsarcasm_file=codecs.open('../../data/nonsarcasm_tweets.txt', encoding='utf-8')

wordSet=set()
#combined word set for sarcasm and non-sarcasm word

for line in sarcasm_file:
    # remove emoticons from sarcasm File
    pattern=r'[\U0001f600-\U0001f650]'
    # remove pattern from sarcasm File
    re.sub(pattern,'',line)
    list1=line.split()
    for word in list1:
        word=word.lower()
        wordSet.add(word)

for line in nonsarcasm_file:
    # remove emoticons from non-sarcasm File
    pattern=r'[\U0001f600-\U0001f650]'
    # remove pattern from non-sarcasm File
    re.sub(pattern,'',line)
    list1=line.split()
    for word in list1:
        word=word.lower()
        wordSet.add(word)


dic={}
#empty Dictionary
word_index=11
#create List for Dictionary
for word in wordSet:
    dic[word_index]= word
    word_index+=1


vocab_file=codecs.open('../../data/vocab.txt','w',encoding='utf-8')
for key in dic:
    vocab_file.write(str(key)+':'+dic[key]+'\n')
