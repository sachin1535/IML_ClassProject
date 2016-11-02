# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import os
# random shuffle
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


model = Doc2Vec.load('./imdb.d2v')

log.info('Sentiment')
train_arrays = numpy.zeros((99895, 100))
train_labels = numpy.zeros(99895)
path  = "/home/mw4vision/Downloads/word2vec-sentiments-master/ML_Project/"
pathTest  = "/home/mw4vision/Downloads/word2vec-sentiments-master/ML_Project/Test"
fd = open("detailInfo.txt",'r');
classCnt = 0;
classRecords = dict();
cnt = 0;
cntc= 0;
for line in fd:
    cntc = 0;
    parts = line.split('\t');
    classRecords[parts[1]] = classCnt;
    for each_sample in range(int(parts[2][:-1])):
        sample_category = parts[1]+"_" + str(cntc)
        cntc = cntc+1;
        print(sample_category);
        train_arrays[cnt] = model.docvecs[sample_category]
        train_labels[cnt] = classCnt;
        cnt = cnt+1;
    classCnt = classCnt+1;
fd.close();
print numpy.max(train_labels)

test_arrays = numpy.zeros((285, 100))
test_labels = numpy.zeros(285)

fd = open("detailInfo.txt",'r');
cnt = 0;
cntc= 0;
print(classRecords);
for line in fd:
    cntc=0;
    parts = line.split('\t');
    cnt = 0;
    classCnt = classRecords[parts[1]];
    for each_sample in range(5):
        sample_category = "TEST_"+parts[1]+"_" + str(cntc)
        cntc = cntc+1;
        test_arrays[cnt] = model.docvecs[sample_category]
        test_labels[cnt] = classCnt;
        cnt = cnt+1;
fd.close();
cntc=0;
for i in range(5):
    sample_category = "TEST_"+parts[1]+"_" + str(cntc)
    cntc = cntc+1;
    test_arrays[cnt] = model.docvecs[sample_category]
    test_labels[cnt] = classRecords["MONEY"];
    cnt = cnt+1;

print(classRecords["MONEY"]);  
print(test_labels);
log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001,multi_class = 'ovr')

#print classifier.score(test_arrays, test_labels)
labels = classifier.predict(test_arrays)
print(len(labels));
cnt=289;
for i in range(len(test_labels)-5,len(test_labels)): 
    if test_labels[i]==labels[cnt]:
        print("correct");
    cnt=cnt+1;
#print labels

print(cntc);
