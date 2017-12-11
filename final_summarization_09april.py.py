
# coding: utf-8

# In[1]:


"""
Created on Sat Apr  2 18:06:25 2016

@author: AS
"""

import re
import logging

import nltk
import os
import re
import json
import unicodedata
import string
import io
import pandas as pd
import textmining

import sklearn
import numpy as np
global collections
import collections
global operator

import operator
global create_tag_image
global make_tags
global LAYOUTS
global get_tag_counts
from nltk import *
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import stem
from sklearn.feature_extraction.text import CountVectorizer
#from pytagcloud import create_tag_image, make_tags, LAYOUTS
#from pytagcloud.lang.counter import get_tag_counts
from nltk import pos_tag
from nltk.tokenize import sent_tokenize


# ## Working with data
# 

# In[2]:

###############################################################################

######## Paragraph segmentation defination
def para(text, doc_id = None):
    
    # removing spaces from the text

    text = re.sub(r'\r', r'',text.strip() )
    res = re.split(r'\n\s*(?:\n|\s\s\s|\t)\s*\n',text)
     # spliting long paragraphs
    if len(res) == 1 and len(text)  > 500 and re.search(r'\.\s*\n',text):
        logging.info('the text has more then 500 charecters and containing no'+
                        'period, newline etc'.format(doc_id))
    res = re.split(r'\n+',text)
   
   #Removng multiple spaces with single space
    res = [re.sub(r'\s+', ' ' , x ) for x in res]
    

   # Make sure number of non whitespaces charecters were unchanged
   
    assert len(re.sub(r'\s', '', text))     == len(''.join([re.sub(r'\s', '', x) for x in res]))
       
    logging.info('Number of paragraph found: {}'.format(len(res)))
       
    return res
  


# In[3]:

##############################################################################

##Input data     
       
text1 = open("7.txt").read()
text1

#############################################################################


# In[4]:

##Pre Processing

#### Sentence Segmentation

sent_seg = sent_tokenize(text1)
len(sent_seg)
sent_seg


# In[5]:

s = unicode(sent_seg)

##############################################################################


# In[6]:

# Convert all characters to Lower case

text_lower=s.lower()
print text_lower

#


# In[44]:

x1 = para(s)
x2 = para(text1)
x3= unicode(x1)
len(text1)


# In[8]:

# Remove all the encoding, escape and special characters
ntext=unicodedata.normalize('NFKD', x3).encode('ascii','ignore')
print ntext


# In[9]:

# Remove all the punctuations from the text
text_nopunc=ntext.translate(string.maketrans("",""), string.punctuation)
print text_nopunc



# In[12]:

remove_list = ['pardpardeftab720sl400sa300partightenfactor0', 'fieldfldinstHYPERLINK',  
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0006fldrslt', 
            'cf5', 'fieldfldinstHYPERLINK', 'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0007fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0009fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0008fldrslt',
            'strokec5','httploginfindlawcomscriptscallawdestcacaapp4th119522htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0005fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0004fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0003fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0002fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0001fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlB0009fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th178192htmlfldrslt','pardpardeftab720sl400qcpartightenfactor0'
            'httploginfindlawcomscriptscallawdestcacaapp4th178192htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th2001454htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th1701530htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th155844htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacal4th271161htmlfldrslt ',
            'httploginfindlawcomscriptscallawdestcacaapp4th120521htmlfldrslt','CalApp4th',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlB0008fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th1041401htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th83460htmlfldrslt ',
            'httploginfindlawcomscriptscallawdestcacaapp4th119988htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacalapp3d144786htmlfldrslt', 'pardpardeftab720sl400qcpartightenfactor0',
            'httploginfindlawcomscriptscallawdestcacaapp4th1721268htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th71240htmlfldrslt'
            ]
            
            
word_list = text_nopunc.split() 
clean = ' '.join([i for i in word_list if i not in remove_list])
clean


# In[16]:

# Convert all characters to Lower case
text_lower=text_nopunc.lower()
print text_lower
word_list1 = text_lower.split() 
s2 = unicode(word_list)
clean1 = ' '.join([i for i in word_list1 if i not in remove_list])


# 

# In[17]:

# Create a stopword list from the standard list of stopwords available in nltk
stop = stopwords.words('english')
print stop


# In[20]:

# Remove all these stopwords from the text
text_nostop=" ".join(filter(lambda word: word not in stop, clean1.split()))
print text_nostop


# In[21]:

# Convert the stopword free text into sentence tokens to enable further processing
sent_tok = sent_tokenize(text_nostop)
print sent_tok


# In[22]:

# Convert the stopword free text into word tokens to enable further processing
tokens = word_tokenize(text_nostop)
print tokens


# In[23]:

# Now, for Lemmatization, which converts each word to it's corresponding lemma, use the Lemmatizer provided by nltk
wnl = nltk.WordNetLemmatizer()
text_lem=" ".join([wnl.lemmatize(t) for t in tokens])
print text_lem


# In[24]:

# Try the default tagger with the Processed Text
pos = pos_tag(word_tokenize(text_lem)) 
print pos


# In[25]:

# Displaying ages and names in the given data
ages = re.findall(r'\d{1,3}',ntext)     # to be modified for age, giving numbers
print(ages)


names = re.findall(r'[A-Z][a-z]*',ntext)
print(names)


# 

# In[26]:

from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest


# In[27]:

class FrequencySummarizer:
  def __init__(self, min_cut=0.1, max_cut=0.9):
    
    
    self._min_cut = min_cut
    self._max_cut = max_cut 
    self._stopwords = set(stopwords.words('english') + list(punctuation))

  def _compute_frequencies(self, word_sent):
   
    freq = defaultdict(int)
    for r in word_sent:
      for word in r:
        if word not in self._stopwords:
          freq[word] += 1
    # frequencies normalization and fitering
    m = float(max(freq.values()))
    for w in freq.keys():
      freq[w] = freq[w]/m
      if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
        del freq[w]
    return freq

  def summarize(self, text, n):
    
    sent_tok = sent_tokenize(x3)
    assert n <= len(sent_tok)
    word_sent = [word_tokenize(r.lower()) for r in sent_tok]
    self._freq = self._compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):
      for w in sent:
        if w in self._freq:
          ranking[i] += self._freq[w]
    sents_idx = self._rank(ranking, n)    
    return [sent_tok[j] for j in sents_idx]

  def _rank(self, ranking, n):
    """ return the n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)
    
    
    
    


# In[32]:

fs = FrequencySummarizer()

y = fs.summarize(x3, 5)

print y


# In[ ]:

mylist = ', '.join(y)
print mylist


# In[36]:

z = para(mylist)
print z


# In[ ]:

y = unicode(y)
s2 = re.sub(r'[^\w\s]','',y)





remove_list = ['pardpardeftab720sl400sa300partightenfactor0', 'fieldfldinstHYPERLINK',  
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0006fldrslt', 
            'cf5', 'fieldfldinstHYPERLINK', 'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0007fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0009fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0008fldrslt',
            'strokec5','httploginfindlawcomscriptscallawdestcacaapp4th119522htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0005fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0004fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0003fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0002fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlA0001fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlB0009fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th178192htmlfldrslt','pardpardeftab720sl400qcpartightenfactor0'
            'httploginfindlawcomscriptscallawdestcacaapp4th178192htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th2001454htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th1701530htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th155844htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacal4th271161htmlfldrslt ',
            'httploginfindlawcomscriptscallawdestcacaapp4th120521htmlfldrslt','CalApp4th',
            'httploginfindlawcomscriptscallawdestcacaapp4thslip2015d068146htmlB0008fldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th1041401htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th83460htmlfldrslt ',
            'httploginfindlawcomscriptscallawdestcacaapp4th119988htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacalapp3d144786htmlfldrslt', 'pardpardeftab720sl400qcpartightenfactor0',
            'httploginfindlawcomscriptscallawdestcacaapp4th1721268htmlfldrslt',
            'httploginfindlawcomscriptscallawdestcacaapp4th71240htmlfldrslt'
            ]
            
            
word_list = s2.split() 
clean = ' '.join([i for i in word_list if i not in remove_list])
clean1 = re.sub(r'\d{1,3}','',clean)
clean2 = re.sub(r'^\b\w+\W\[a-z]\{15,100}\b','',clean1)


# In[ ]:




# In[ ]:




# In[40]:

clean
clean1


# In[45]:

clean2
len(clean2)


# In[52]:

clean1.replace('httploginfindlawcomscriptscallawdestcacaappthhtmlfldrslt', '')
names = re.sub("httploginfindlawcomscriptscallawdestcacaappthhtmlfldrslt",'',clean1)
print y.strip()
print "Summary Length %s" % len(clean2)
print "Original Length %s" % len(x3)


# In[ ]:



