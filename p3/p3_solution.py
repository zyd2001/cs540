from collections import Counter, OrderedDict
from itertools import product
# import matplotlib.pyplot as plt
from random import choices

import numpy as np
import string
import sys
import re
import math

# in this piece of code, I leave out a bunch of thing for you to fill up modify.
# The current code may run into a ZeroDivisionError. Thus, you need to add Laplace first.
'''
Todo: 
1. Laplace smoothing
2. Naive Bayes prediction
3. All the output.

'''


with open('script.txt', encoding='utf-8') as f:
    data = f.read()
# len(data)

data = data.lower()
data = data.translate(str.maketrans('', '', string.punctuation))
data = re.sub('[^a-z]+', ' ', data)
data = ' '.join(data.split(' '))

allchar = ' ' + string.ascii_lowercase

unigram = Counter(data)
unigram_prob = {ch: round((unigram[ch]) / (len(data)), 4) for ch in allchar}

uni_list = [unigram_prob[c] for c in allchar]

def ngram(n):
    # all possible n-grams
    d = dict.fromkeys([''.join(i) for i in product(allchar, repeat=n)],0)
    # update counts
    d.update(Counter([''.join(j) for j in zip(*[data[i:] for i in range(n)])]))
    return d

bigram = ngram(2)  # c(ab)
bigram_prob = {c: bigram[c] / unigram[c[0]] for c in bigram}  # p(b|a)


trigram = ngram(3)
trigram_prob = {c: (trigram[c]) / (bigram[c[:2]]) for c in trigram}


def gen_bi(c):
    w = [bigram_prob[c + i] for i in allchar]
    return choices(allchar, weights=w)[0]
    

def gen_tri(ab):
    w_tri = [trigram_prob[ab + i] for i in allchar]
    return choices(allchar, weights=w_tri)[0]   


def gen_sen(c, num):
    res = c + gen_bi(c)
    for i in range(num - 2):
        if bigram[res[-2:]] == 0:
            t = gen_bi(res[-1])
        else:
            t = gen_tri(res[-2:])
        res += t
    return res


example_sentence = gen_sen('h', 100)


with open('script2.txt', encoding='utf-8') as f:
    young = f.read() 

dict2 = Counter(young)
likeli = [dict2[c] / len(young) for c in allchar]
post_young = [round(likeli[i] / (likeli[i] + uni_list[i]), 4) for i in range(27)]
post_hugh = [1 - post_young[i] for i in range(27)]

print(post_young)
