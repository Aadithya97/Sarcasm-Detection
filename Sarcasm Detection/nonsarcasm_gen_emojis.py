# Count emojis from tweets
# Count LOL, ROFL and LMAO

import re
import codecs
import csv

nonsarcasm = codecs.open('../../data/nonsarcasm_tweets.txt', encoding='utf-8')
count=[]
target=[]
laughexp=[]

for line in nonsarcasm:
    target.append(line)
    count.append(len(re.findall(u'[\U0001f600-\U0001f650]', line)))
    part1=re.compile(r'(\blols?z?o?\b)+?',re.I)
    #For Lols
    part2=re.compile(r'(\brofl\b)+?',re.I)
    #For rofl
    part3=re.compile(r'(\blmao\b)+?',re.I)
    #For lmao
    laughexp.append(len(re.findall(part1,line)) + len(re.findall(part2,line)) + len(re.findall(part3,line)))
    try:
        next(nonsarcasm)
    except:
        break


with open('../../data/nonsarcasm_emoji.csv','w') as csvfile:
    x=csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    x.writerow(['emoji_count','laughter_exp_count'])
    for i in range(len(target)):
        x.writerow([count[i],laughexp[i]])

csvfile.close()
