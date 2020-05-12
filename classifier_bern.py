import numpy as np
import numpy.linalg as npla
import math
import os


# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

class Nbclassifier:
    wordChoice = None
    probDict_0 = None
    probDict_1 = None
    prior_0 = None
    prior_1 = None

    def __init__(self, _wordChoice, _smooth):
        self.wordChoice = _wordChoice
        self.probDict_0 = self.format_wordChoice()
        self.probDict_1 = self.format_wordChoice()
        self.probDict = self.format_wordChoice()
        self.smooth = _smooth
    def format_wordChoice(self):
        wordDict = {}
        for word in self.wordChoice:
            wordDict.update({word : 0})
        return wordDict
    

    def update_frequency(self, wordDict, lineData):
        word = ''
        encountered = {}
        for c in lineData:
            if c == ' ':
                if wordDict.get(word) is not None:
                    if encountered.get(word) is None:
                        wordDict[word] = wordDict[word] + 1
                        encountered.update({word: 1})
                word = ''
            elif c == ',':
                break
            else:
                word = word + c

    def train(self, data):
        freqDict_0 = self.format_wordChoice()
        freqDict_1 = self.format_wordChoice()
        freqDict   = self.format_wordChoice()
        count_0 = 0
        count_1 = 0
        data_y = []
        prog = 0
        #print('training on dataset')
        for line in data:
            prog += 1
            if line[-2] == '0':
                self.update_frequency(freqDict_0, line)
                count_0 += 1
                data_y.append(0)
            else:
                self.update_frequency(freqDict_1, line)
                count_1 += 1
                data_y.append(1)
            self.update_frequency(freqDict, line)
            #if prog%2000 == 0:
            #    print(prog, 'in', len(data))
        prog = 0
        #print('formatting probability')
        #print('count 0', count_0, 'count 1', count_1)
        for key in freqDict_0:
            prog += 1
            self.probDict_0[key] = freqDict_0[key] / count_0
            self.probDict_1[key] = freqDict_1[key] / (count_1 - self.smooth*230)
            self.probDict[key] = freqDict[key] / (count_0 + count_1)
            #if prog%10000 == 0:
            #    print(prog, 'in', len(freqDict_0))
        self.prior_0 = count_0 / (count_0 + count_1) 
        self.prior_1 = count_1 / (count_0 + count_1)
        pred = self.classify(data)
        conf = np.array(pred) - np.array(data_y)
        unique, counts = np.unique(conf, return_counts=True)
        #print('-1 fake neg, 1 fake pos',dict(zip(unique, counts)))
        acc = 1 - np.count_nonzero(conf)/conf.shape[0]
        #print("sum of ground truth", np.sum(data_y))
        #print("sum of predicted   ", np.sum(pred))
        return acc

    def find_prob(self, wordDict, probDict, prior):
        val = prior
        '''for key in probDict:
            if wordDict.get(key) is not None:
                val += np.log(probDict[key] + self.smooth * self.probDict[key])
            else:
                val += np.log(1 - probDict[key] + self.smooth * self.probDict[key])
        '''
        for key in wordDict:
            if probDict.get(key) is not None:
                val += np.log(probDict[key]*(wordDict[key] ** 1.5) + self.smooth * self.probDict[key])
        return val
        
    
    def classify(self, data):
        reverse = False
        next_reverse = False
        predicted = []
        word = ''
        prog = 0
        #print('classifying')
        for lineData in data:
            prog += 1
            wordlist = []
            worddict = {}
            for c in lineData:
                if c == ' ':
                    if worddict.get(word) is None:
                        worddict.update({word:1})
                    else:
                        worddict[word] = worddict[word] + 1
                    word = ''
                elif c == ',':
                    break
                else:
                    word = word + c
            prob_0 = self.find_prob(worddict, self.probDict_0, self.prior_0)
            prob_1 = self.find_prob(worddict, self.probDict_1, self.prior_1)
            if(prob_0 > prob_1 and not reverse) or (prob_0 < prob_1 and reverse):
                predicted.append(0)
            else:
                predicted.append(1)
                
            #if prog%2000 == 0:
            #    print(prog, 'in', len(data))
        return predicted

    def validate(self, tdata):
        tdata_y = []
        for line in tdata:
            if line[-2] == '0':
                tdata_y.append(0)
            else:
                tdata_y.append(1)
        pred = self.classify(tdata)
        
        conf = np.array(pred) - np.array(tdata_y)
        cont0 = 0
        cont1 = 0
        for i in range(len(conf)):
            if(conf[i] == -1 and cont0 < 20):
                #print('fake neg', tdata[i])
                cont0 += 1
            elif(conf[i] == 1 and cont1 < 0):
                #print('fake pos', tdata[i])
                cont1 += 1
        unique, counts = np.unique(conf, return_counts=True)
        #print('-1 fake neg, 1 fake pos',dict(zip(unique, counts)))
        acc = 1 - np.count_nonzero(conf)/conf.shape[0]
        return acc



    
wordListPurity = np.load('wordListPurity.npy')

fpath = 'training.txt'
if not os.path.exists(fpath):
    print('file no exist')
f = open(fpath, encoding = 'utf-8')
data = f.readlines()

tpath = 'testing.txt'
if not os.path.exists(fpath):
    print('file no exist')
t = open(tpath, encoding = 'utf-8')
tdata = t.readlines()

myChoice = ['good', 'well', 'awesome']
maxi = int(wordListPurity.shape[0] * 0.8)
for i in range(maxi):
    if(watchlist.get(wordListPurity[i, 0]) is not None):
        print(wordListPurity[i,0], purity)
    purity = float(wordListPurity[i,1])
    hig_filt = (1 + 0.18 * np.log(1 + i/maxi))
    low_filt = (1 - 0.20 * np.log(1 + i/maxi))
    if purity > 0.64 or purity < 0.43:
        myChoice.append(wordListPurity[i, 0])
print('mychoicelen', len(myChoice))
myClassifier = Nbclassifier(myChoice, 3)

acc = myClassifier.train(data)
print('accuracy', acc)
acc = myClassifier.validate(tdata)
print('accuracy', acc)

