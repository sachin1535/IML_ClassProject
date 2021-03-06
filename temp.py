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

log.info('Sentiment')
class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
	return self.sentences
        

log.info('source load')
path = '/home/mw4vision/Downloads/word2vec-sentiments-master/ML_Project/'
pathTest = '/home/mw4vision/Downloads/word2vec-sentiments-master/ML_Project/Test/'
sources = dict()
for fn in os.listdir(path):
	if fn.endswith('.txt'):
		fileparts = (fn.split('.'))[0]
		sources[str(path+fn)] = str(fileparts.upper());
#print(sources);
for fn in os.listdir(pathTest):
    if fn.endswith('.txt'):
        fileparts = (fn.split('.'))[0]
        sources[str(pathTest+fn)] = str(fileparts.upper());

print(sources);
log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

log.info('D2V')
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=5)                     #vary size=200
model.build_vocab(sentences.to_array())

log.info('Epoch')
for epoch in range(30):                                            #vary this
	log.info('EPOCH: {}'.format(epoch))
	model.train(sentences.sentences_perm())

log.info('Model Save')
model.save('./imdb.d2v')
