#import the appropriate libraries
import math
#numpy is used for mathematical functions
import numpy as np
#csv is used for csv file processing
import csv
from keras.models import Sequential
from keras.layers import Dense
import random
import math
from sklearn.metrics import precision_recall_curve
#create the input based on the final dataset
def read_input():
    A = [];
    b = [];
    C = [];
        #non-sarcasm based input
    with open('../../data/nonsarcasm_final_features.csv', 'rb') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        init = 1
        for row in file:
            if(init == 1):
                init = 0
                continue
            temp = [float(i) for i in row]
            A.append(temp[0:10])
            b.append(0)

        #sarcasm based input
    with open('../../data/sarcasm_final_features.csv', 'rb') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        init = 1
        for row in file:
            if(init == 1):
                init = 0
                continue
            temp = [float(i) for i in row]
            A.append(temp[0:10])
            b.append(1)
    A = np.asarray(A)
    b = np.asarray(b)

    D = np.c_[A, b]
    np.random.shuffle(D)

    X = D[:,0:10]
    y = D[:,10]
    return X,y

[A,b] = read_input()
#At this stage, the input is succesfully read

num_rows, num_cols = A.shape
n_train = int(num_rows*0.80)
#use the 80:20 rule for dividing the dataset into training data and test data
X = A[0:n_train,:]
Y = b[0:n_train]

X_valid = A[n_train:num_rows,:]
Y_valid = b[n_train:num_rows]
#At this stage, the data is processed

#set the number of nodes for each layer
n_input = 10   #input
n1=5           #hidden
n2=1           #output

model = Sequential()
model.add(Dense(output_dim = n1, input_dim=n_input, init='uniform', activation='relu'))
model.add(Dense(output_dim = n2, init='uniform', activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #configure learning
model.fit(X, Y, nb_epoch=10, batch_size=10) # trains for fixed epochs
scores = model.evaluate(X_valid, Y_valid)  #accuracy
test_classes = model.predict_classes(X_valid)  #generate output predictions batch by batch
 

p_label = test_classes
p_actual = Y_valid
tp,fp,tn,fn=0.0,0.0,0.0,0.0
for predicted,actual in zip(p_label,p_actual):
    if(predicted==actual and predicted==1): tp+=1
    elif(predicted==actual and predicted==0):tn+=1
    elif(predicted==0):fn+=1
    else:fp+=1

precision = float(tp)/(tp+fp)
recall = float(tp)/(tp+fn)
fscore = 2*precision*recall/(precision+recall)

print ("\n")
print ("the precision is: ",precision)
print ("the recall is: ",recall)
print ("the f-measure is: ",fscore)

print ("the final accuracy is: ",scores[1])