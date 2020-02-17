import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence
#return set of words appear for more than once
def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    if n<=2:
        sequence=['START']+list(sequence)+['STOP']
    else:
        sequence=['START']*(n-1)+list(sequence)+['STOP']
    ans=[]
    for i in range(n-1,len(sequence)):
        temp=[]
        for j in range(i-n,i):
            temp.append(sequence[j+1])
        ans.append(tuple(temp))
    return ans


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int)  # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        for sentence in corpus:
            unigram=get_ngrams(sentence,1)
            for uni in unigram:
                self.unigramcounts[uni]+=1
            bigram=get_ngrams(sentence,2)
            self.bigramcounts[('START', 'START')] += 1
            for big in bigram:
                self.bigramcounts[big]+=1
            trigram=get_ngrams(sentence,3)
            for tri in trigram:
                self.trigramcounts[tri]+=1
        self.total_words=sum(self.unigramcounts.values())
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        return float(self.trigramcounts[trigram])/self.bigramcounts[trigram[:2]] if self.bigramcounts[trigram[:2]]!=0 else 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return float(self.bigramcounts[bigram])/self.unigramcounts[bigram[:1]]

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        return float(self.unigramcounts[unigram]/self.total_words)

    def generate_sentence(self,t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        len=0
        sentence=[]
        first,second='START','START'
        while len<20:
            alltrigrams = [eachtri for eachtri in self.trigramcounts.keys() if eachtri[:2] == (first, second)]
            probabilities = [self.raw_trigram_probability(trigram) for trigram in alltrigrams]
            flag=random.uniform(0,sum(probabilities))
            for i,word in enumerate(alltrigrams):
                flag-=probabilities[i]
                if flag<0:
                    third=word[2]
                    sentence.append(third)
                    len+=1
                    first = second
                    second = third
                    if word[2]=='STOP':
                        return sentence
                    break
        return sentence

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        proba = lambda1 * self.raw_trigram_probability(trigram) + \
              lambda2 * self.raw_bigram_probability(trigram[1:]) + \
              lambda3 * self.raw_unigram_probability(trigram[2:])
        return proba

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        tri_grams=get_ngrams(sentence,3)
        ans=0
        for tri_gram in tri_grams:
            ans+=math.log2(self.smoothed_trigram_probability(tri_gram))
        return ans

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l=0
        total_words=0
        for sentence in corpus:
            l+=self.sentence_logprob(sentence)
            total_words+=len(sentence)
        return 2**(-l/total_words)



def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0

        for f in os.listdir(testdir1):  # high labels
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            correct += int(pp1 < pp2)

        for f in os.listdir(testdir2):  # low labels
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            total += 1
            correct += int(pp1 > pp2)

        return float(correct) / total

if __name__ == "__main__":
    print("test function: get_ngrams")
    print(get_ngrams(['natural', 'language', 'processing'], 1))
    print(get_ngrams(['natural', 'language', 'processing'], 2))
    print(get_ngrams(['natural', 'language', 'processing'], 3))
    print("***************************")

    model = TrigramModel('brown_train.txt')
    print("model.trigramcounts[('START','START','the')]: ",model.trigramcounts[('START','START','the')])
    print("model.bigramcounts[('START','the')]: ",model.bigramcounts[('START','the')])
    print("model.unigramcounts[('the',)]: ",model.unigramcounts[('the',)])
    dev_corpus = corpus_reader('brown_test.txt', model.lexicon)
    pp = model.perplexity(dev_corpus)
    print('perplexity for brown_test:',pp)
    dev_corpus = corpus_reader('brown_train.txt', model.lexicon)
    pp = model.perplexity(dev_corpus)
    print('perplexity for brown_train:',pp)
    print('random sentence:',model.generate_sentence(20))
    print('random sentence:',model.generate_sentence(20))
    print('random sentence:',model.generate_sentence(20))
    # put test code here...
    #model = TrigramModel(sys.argv[1])
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt.
    #Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    print(acc)

