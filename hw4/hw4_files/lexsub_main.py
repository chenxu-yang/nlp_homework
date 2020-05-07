#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = []
    l1=wn.lemmas(lemma,pos=pos)
    for l in l1:
        lex=l.synset().lemmas()
        for le in lex:
            word=le.name()
            if word not in possible_synonyms and word !=lemma:
                if '_' in word:
                    word.replace('_',' ')
                possible_synonyms.append(word)


    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    lemma=context.lemma
    pos=context.pos
    l1=wn.lemmas(lemma,pos=pos)
    record_req={}
    for l in l1:
        lex=l.synset().lemmas()
        for le in lex:
            word=le.name()
            if word != lemma:
                if word not in record_req:
                    record_req[word]=le.count()
                else:
                    record_req[word]+=le.count()
    return max(record_req,key=record_req.get)

def wn_simple_lesk_predictor(context):
    lemma=context.lemma
    pos=context.pos
    total_sentence=set(context.left_context+context.right_context)
    sentence=[]
    def_reference={}
    for word in total_sentence:
        if word not in stopwords.words('english'):
            sentence.append(PorterStemmer().stem(word))
    sentence=set(sentence)
    l1=wn.lemmas(lemma,pos=pos)
    for l in l1:
        lex=l.synset().lemmas()
        for le in lex:
            s=le.synset()
            definition=word_tokenize(s.definition())+s.examples()
            for hy in s.hypernyms():
                definition+=word_tokenize(hy.definition())
                for ex in hy.examples():
                    definition+=word_tokenize(ex)
            definition=set([WordNetLemmatizer().lemmatize(word) for word in definition if word not in stopwords.words('english')])
            overlap_count=len(definition.intersection(sentence))
            if overlap_count:
                word=le.name()
                if word !=lemma:
                    if word not in def_reference:
                        def_reference[word]=overlap_count
                    else:
                        def_reference[word]+=overlap_count
    if not def_reference:
        return wn_frequency_predictor(context)
    else:
        return max(def_reference,key=def_reference.get)


   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context):
        possible_synonyms=get_candidates(context.lemma,context.pos)
        max_sim=0
        ans=None
        for sy in possible_synonyms:
            if sy in self.model.wv.vocab:
                this_sim=self.model.similarity(context.lemma,sy)
                if this_sim>max_sim:
                    max_sim=this_sim
                    ans=sy
        return ans
    def predict_nearest_with_context(self, context):
        total_sentence = set(context.left_context[-5:] + context.right_context[0:5])
        sentence = []
        vector_sum=self.model.wv[context.lemma]
        for word in total_sentence:
            if word not in stopwords.words('english'):
                sentence.append(word)
        for word in sentence:
            if word in self.model.wv.vocab:
                vector_sum=vector_sum+self.model.wv[word]
        possible_synonyms=get_candidates(context.lemma,context.pos)
        max_sim=0
        ans=None
        for sy in possible_synonyms:
            if sy in self.model.wv.vocab:
                this_sim=cos(self.model.wv[sy],vector_sum)
                if this_sim>max_sim:
                    max_sim=this_sim
                    ans=sy
        return ans
    def own_predict_nearest_with_context(self, context):
        total_sentence = set(context.left_context + context.right_context)
        sentence = []
        vector_sum=self.model.wv[context.lemma]
        for word in total_sentence:
            if word not in stopwords.words('english'):
                sentence.append(word)
        for word in sentence:
            if word in self.model.wv.vocab:
                vector_sum=vector_sum+self.model.wv[word]
        possible_synonyms=get_candidates(context.lemma,context.pos)
        max_sim=0
        ans=None
        for sy in possible_synonyms:
            if sy in self.model.wv.vocab:
                this_sim=cos(self.model.wv[sy],vector_sum)
                if this_sim>max_sim:
                    max_sim=this_sim
                    ans=sy
        
        return ans

def cos(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
if __name__=="__main__":
    # print(get_candidates('slow', 'a'))
    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = predictor.own_predict_nearest_with_context(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
