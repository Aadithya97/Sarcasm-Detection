# Used to insert gaps in sarcasm_tweets.txt (required for certain codes)
import re
import codecs
import csv
fileInput = open('../../data/sarcasm_tweets.txt')
fileOutput = open('../../data/sarcasm_tweets_gaps.txt','wb')

for line in fileInput:
	fileOutput.write(line)
	#Insert Gaps
	fileOutput.write('\n') 
	
fileOutput.close()

