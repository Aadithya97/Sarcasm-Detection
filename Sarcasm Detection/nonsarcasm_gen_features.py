# Generate primary first level features
# (+ve word count) (-ve word count) (overall polarity) (times change in pol) (largest + cont) (largest - cont) (capital letters) (punctuation)

import re,csv
import codecs

#Read Data from Non Sarcasm Text  
file3 = codecs.open('../../data/nonsarcasm_tweets.txt', encoding='utf-8')
#Result of computation in nonsarcasm_gen_features.csv
csvfile = open('../../data/nonsarcasm_gen_features.csv','wb')
csvwriter =csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

# print('(+ve word count) (-ve word count) (overall polarity) (times change in pol) (largest + cont) (largest - cont) (capital letters) (punctuation)')
csvwriter.writerow(['(+ve word count) ','(-ve word count)' ,'(overall polarity)' ,'(times change in pol)' ,'(largest + cont)' ,'(largest - cont)' ,'(capital letters)' ,'(punctuation)'])
file2 = open('../../data/nonsarcasm_tweets_clean.txt')

pattern=re.compile(r'\[(\-?[0-9])\]')
capital_count=[]
puncuation_count=[]
q=0
for line in file3 :
    puncuation_count.append(line.count('!')+line.count('?'))
    for line1 in range(len(line)-2):
        if line[line1]=='.' and line[line1+2]=='.':
            puncuation_count[-1]=puncuation_count[-1]+1

# print len(puncuation_count)
for line,inde in zip(file2,range(len(puncuation_count))):
    target= line.split(' ')
    ls=[]
    for line1 in target :
        try:
            ls.append(int(pattern.findall(line1)[0]))
        except:
            continue
    pc,nc=0,0
    lsp,lsn=[],[]

    lsp = list(filter(lambda(x):x>0 , ls ))
    lsn= list(filter(lambda(x):x<0 , ls ))
    change_pol,previous=0,0
    for line1 in ls :
        if(line1==0):
            continue
        if(previous==0 and line1>0):previous=1
        elif(previous==0 and line1<0):previous=-1
        elif(line1>0 and previous<0 or line1<0 and previous>0):
            change_pol=change_pol+1
            previous=-1*previous


    neg_max,pos_max=0,0
    current=0
    previous=0
    for line1 in ls :
        if(line1==0):
            continue
        if(previous==0 and line1>0):
            previous=1
            current=1
        elif(previous == 0 and line1<0):
            previous = -1
            current = 1
        elif(line1>0 and previous>0 or line1<0 and previous<0):current=current+1
        elif(line1>0 and previous<0 or line1<0 and previous>0):
            if(previous<0):neg_max=max(neg_max,cur)
            else:pos_max=max(pos_max,cur)
            previous= -1*previous
            current=1
    else:
        if(previous<0):neg_max=max(neg_max,current)
        elif(previous>0):pos_max=max(pos_max,current)

    capital_count.append(sum(list(map(lambda x : x.isupper(),list(line))))- line.count('NAME')*4 - line.count('HYPERLINK')*9)

    try:
        q+=1
        csvwriter.writerow([len(lsp),len(lsn),sum(lsp)+sum(lsn),change_pol,pos_max,neg_max,capital_count[-1],puncuation_count[inde]])
    except Exception as e:
        pass

# print q
csvfile.close()
