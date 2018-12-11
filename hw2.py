#!/usr/bin/env python
# coding:utf-8
import nltk
#nltk.download('punkt') 
#nltk.download('stopwords')
import json
import os
import math
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from collections import OrderedDict
import numpy as np


# the dir of the resource txt
path_resource = '/Users/mileszero/Desktop/3_1/TEXT_MINING/hw2/IRTM'
path_res = '/Users/mileszero/Desktop/3_1/TEXT_MINING/hw2/res'


def readFile(filename):
	fopen = open(path_resource + '/' + filename)
	fileContext = fopen.read()
	fopen.close()
	return fileContext

def getTerm(txt):
	word_tokens = word_tokenize(txt) 
	# lowercase
	word_tokens =[word.lower() for word in word_tokens if word.isalpha()]
	# stopwords removal
	stop_words = set(stopwords.words('english')) 
	filtered_sentence = [w for w in word_tokens if not w in stop_words] 
	filtered_sentence = [] 
	for w in word_tokens:
		if w not in stop_words: 
			filtered_sentence.append(w) 
	# porter algorithm
	porter = nltk.PorterStemmer()
	porter_sentence = []
	for x in filtered_sentence:
		porter_sentence.append(porter.stem(x))
	return porter_sentence

def ListToDict(L = []):
	counts = dict()
	for i in L:
		counts[i] = counts.get(i, 0) + 1
	return counts

def writeDict(outFilename):
	with open(outFilename, 'w') as f:
		f.write("t_index  term    df\n")
	for i in range(len(dict2)):
		with open(outFilename, 'a') as f:
			f.write(str(i + 1) + "    " + str(dictKeyList[i]) + "  " + str(dict2.get(dictKeyList[i])) + "\n")

def writeDoc():
	for j in range(docNum):
		fn = str(j + 1) + '.txt'
		with open(path_res + '/' + fn, "w") as f:
			f.write(str(int(termNum_inDoc[j])) + "\n")
			f.write("t_index    tf-idf \n")
		for i in range(termNum):
			if (tfidf_Norm[i][j] != 0):    
				with open(path_res + '/' + fn, 'a') as f:
					#f.write((str(dictKeyList.index(word[j]) + 1)) + "      " +  str(weight[i][j]) + "  " + str(word[j]) +"\n")
					f.write((str(i + 1)) + "      " +  str(tfidf_Norm[i][j]) + "\n")

def cal_cos_sim (x, y):
	res = 0.0
	for i in range(termNum):
		res += tfidf_Norm[i][x] * tfidf_Norm[i][y]
	print res

# read the text one by one and combine them up
doc = []
docNum = 0    # count the documents number
#path_list = os.listdir(path_resource)
#sorted_path_list = path_list.sort()
for filename in sorted(os.listdir(path_resource)):
    if filename.endswith('.txt'):
		# print filename
		docNum += 1
		eachTxt = readFile(filename)
		# i dont think number is a kind of term, so i remove number in all text
		eachTxt2 = ''.join([i for i in eachTxt if not i.isdigit()])
		doc.append(eachTxt2)


# print dictionary
txt = ' '.join(doc)    # concat
term1 = getTerm(txt)    # get the raw terms
dict1 = ListToDict(term1)    # convert list to dict
dict2 = OrderedDict(sorted(dict1.items(), key=lambda t: t[0]))    # order dictionary
termNum = len(dict2)    # count the term number in dictionary
dictKeyList = dict2.keys()
#print dictKeyList
writeDict ('dictionary.txt')    # write dictionary
#print termNum

dft = np.zeros(termNum)
idft = np.zeros(termNum)
tfidf = np.zeros((termNum,docNum))
tfidf_Norm = np.zeros((termNum,docNum))
tf = np.zeros((termNum, docNum))
termNum_inDoc = np.zeros(docNum)

# tf
for i in range(docNum):
	tempTerm = getTerm(doc[i])
	for j in range(len(tempTerm)):
		if tempTerm[j] in dictKeyList:
			#print tempTerm[j], " ", dictKeyList.index(tempTerm[j])
			tf[dictKeyList.index(tempTerm[j])][i] += 1


for i in range(termNum):
	#print tf[i]
	for j in range(docNum):
		if tf[i][j] != 0:
			dft[i] += 1
	idft[i] = math.log10(docNum / dft[i])

# tfidf before normalize
for i in range(termNum):
	for j in range(docNum):
		tfidf[i][j] = tf[i][j] * idft[i]


# tfidf after normalize
for j in range(docNum):
	square_sum = 0.0
	for i in range(termNum):
		square_sum += math.pow(tfidf[i][j],2)
	vector_length = math.sqrt(square_sum)
	for i in range(termNum):
		tfidf_Norm[i][j] = tfidf[i][j] / vector_length
		if tfidf_Norm[i][j] != 0:
			termNum_inDoc[j] += 1

''' check normalization, all doc vector length is 1
for j in range(docNum):
	sum = 0.0
	for i in range(termNum):
		sum += math.pow(tfidf_Norm[i][j],2)
	print sum

'''

# write each doc tf-idf
writeDoc()

# calculate the cosine similarity
cal_cos_sim(1,0)












