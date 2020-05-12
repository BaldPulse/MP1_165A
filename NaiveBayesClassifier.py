import numpy as np
import numpy.linalg as npla
import math
import os
import sys
import time

#-----------------------start of preprocessor------------------------
def preprocessor(data, upperCutoff, lowerCutoff, freqCutoff, wChoice):
    '''
    preprocessor, generates wordchoice beased on corpus data
    input:
        data: corpus data, list of lines
        upperCutoff: upper cutoff of purity
        lowerCutoff: lower cutoff of purity
        freqCutoff: upper cutoff of word frequency rank
        wChoice: list of word choice, can contain custom choice
    output:
        wChoice: list of word choice to be used in Nbclassifier
    '''
    #get word frequency wrt class
    wordBag = {}
    nil_count = 0
    one_count = 0
    for line in data:
        word = ''
        label = int(line[-2])
        line_dict = {}
        for c in line:
            if c == ' ':
                if wordBag.get(word) is None:
                    wordBag.update({word : [1, label]})
                else:
                    wordBag[word][0] = wordBag.get(word)[0] + 1
                    wordBag[word][1] = wordBag.get(word)[1] + label
                word = ''
                if(label == 1):
                    one_count += 1
                else:
                    nil_count += 1
            elif c == ',':
                break
            else:
                word = word + c
    #create sorted based on frequency            
    sortedWordList = np.array(sorted(wordBag.items(), key=lambda item: item[1][0], reverse=True))
    word_count = np.array([nil_count, one_count])
    sortedWordBag = dict(sortedWordList)
    nil_arr = []
    one_arr = []
    labels = []
    wordListPurity = []
    #create purity based on sorted frequency
    for key in sortedWordBag:
        nil_pct = (sortedWordBag[key][0]-sortedWordBag[key][1])/word_count[0]
        one_pct = sortedWordBag[key][1]/word_count[1]
        nil_portion = nil_pct/(one_pct + nil_pct)
        wordListPurity.append((key , nil_portion))
    wordListPurity = np.array(wordListPurity)
    #make wordchoice based on frequency and purity
    maxi = int(wordListPurity.shape[0] * freqCutoff)
    for i in range(maxi):
        purity = float(wordListPurity[i,1])
        hig_filt = (1 + 0.18 * np.log(1 + i/maxi))
        low_filt = (1 - 0.20 * np.log(1 + i/maxi))
        if purity > upperCutoff or purity < lowerCutoff:
            wChoice.append(wordListPurity[i, 0])
    return wChoice

#--------------end of preprocessor---------------------

#--------------start of Nbclassifier-------------------
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
        prog = 0
        for key in freqDict_0:
            prog += 1
            self.probDict_0[key] = freqDict_0[key] / count_0
            self.probDict_1[key] = freqDict_1[key] / (count_1 - self.smooth*230)
            self.probDict[key] = freqDict[key] / (count_0 + count_1)
        self.prior_0 = count_0 / (count_0 + count_1) 
        self.prior_1 = count_1 / (count_0 + count_1)
        return 0

    def find_prob(self, wordDict, probDict, prior):
        val = prior
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
                cont0 += 1
            elif(conf[i] == 1 and cont1 < 0):
                cont1 += 1
        unique, counts = np.unique(conf, return_counts=True)
        acc = 1 - np.count_nonzero(conf)/conf.shape[0]
        return acc, pred


#--------------------end of Nbclassifier----------------------
    

if __name__ == "__main__":
    fpath = 'training.txt'
    tpath = 'testing.txt'
    if(len(sys.argv) == 3):
        fpath = sys.argv[1]
        tpath = sys.argv[2]

    if not os.path.exists(fpath):
        print('file no exist')
    f = open(fpath, encoding = 'utf-8')
    data = f.readlines()

    if not os.path.exists(fpath):
        print('file no exist')
    t = open(tpath, encoding = 'utf-8')
    tdata = t.readlines()

    myChoice = ['good', 'well', 'awesome']
    wordChoice = preprocessor(data, upperCutoff = 0.64, lowerCutoff = 0.43, freqCutoff = 0.8, wChoice = myChoice)
    myClassifier = Nbclassifier(wordChoice, 3)
    start_time = time.process_time()
    myClassifier.train(data)
    trainTime = time.process_time() - start_time
    start_time = time.process_time()
    testAcc, testPred = myClassifier.validate(tdata)
    testTime = time.process_time() - start_time
    trainAcc, _ = myClassifier.validate(data)
    for c in testPred:
        print(c)
    print(trainTime, 'seconds (training)')
    print(testTime, 'seconds (labeling)')
    print(trainAcc, '(training)')
    print(testAcc, '(testing)')
