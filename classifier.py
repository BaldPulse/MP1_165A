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
    featureSize = None
    mean_0 = None
    mean_1 = None
    mean   = None
    var_0 = None
    var_1 = None
    var   = None
    bad_vector_count = None
    
    def __init__(self, _wordChoice):
        self.wordChoice = _wordChoice
        self.featureSize = len(_wordChoice)
        self.mean_0 = np.zeros(self.featureSize, dtype = 'float')
        self.mean_1 = np.zeros(self.featureSize, dtype = 'float')
        self.mean   = np.zeros(self.featureSize, dtype = 'float')
        self.var_0 = np.zeros(self.featureSize, dtype = 'float')
        self.var_1 = np.zeros(self.featureSize, dtype = 'float')
        self.var   = np.zeros(self.featureSize, dtype = 'float')
        self.bad_vector_count = 0

    def format_wordChoice(self):
        wordDict = {}
        for word in self.wordChoice:
            wordDict.update({word : 0})
        return wordDict
    
    def format_feature(self, wordDict, lineData):
        x = []
        word = ''
        for c in lineData:
            if c == ' ':
                if wordDict.get(word) is not None:
                    wordDict[word] = wordDict[word] + 1
                word = ''
            elif c == ',':
                break
            else:
                word = word + c
        for w in wordDict:
            x.append(wordDict[w])
        x = np.array(x)
        if(sum(x) != 0):
            x = x / npla.norm(x)
        else:
            self.bad_vector_count += 1
        return x
    
    def train(self, data):
        data_x_0 = []
        data_x_1 = []
        data_y = []
        count_0 = 0
        count_1 = 0
        prog = 0
        for line in data:
            prog += 1
            wd = self.format_wordChoice()
            x = self.format_feature(wd, line)
            if line[-2] == '0':
                data_x_0.append(x)
                count_0 += 1
                data_y.append(0)
            else:
                data_x_1.append(x)
                count_1 += 1
                data_y.append(1)
            if prog%100 == 0:
                print(prog, 'in', len(data))
        data_x_0 = np.array(data_x_0)
        data_x_1 = np.array(data_x_1)
        data_x   = np.append(data_x_0, data_x_1, axis = 0)
        self.mean_0 = np.average(data_x_0, axis = 0)
        self.mean_1 = np.average(data_x_1, axis = 0)
        self.mean = np.average(data_x, axis = 0)
        if data_x_0.shape[0] <= 1 or data_x_1.shape[0] <= 1:
            print('bad training data')
            return -1
        
        for f in data_x_0:
            self.var_0 += (f - self.mean_0) * (f - self.mean_0)
        for f in data_x_1:
            self.var_1 += (f - self.mean_1) * (f - self.mean_1)
        for f in data_x:
            self.var   += (f - self.mean)   * (f - self.mean) 
        
        self.var_0 = self.var_0 / (data_x_0.shape[0] - 1)
        self.var_1 = self.var_1 / (data_x_1.shape[0] - 1)
        self.var   = self.var     / (data_x.shape[0]   - 1)
        self.prior_0 = count_0 / (count_0 + count_1)
        self.prior_1 = count_1 / (count_0 + count_1)


        #return accuracy data
        #print('groundtruth', data_y)
        pred = self.classify(data)
        acc = 1 - np.average(np.abs(np.array(pred) - np.array(data_y)))
        return acc

    def gaussian(self, x, var, mu):
        val = 1.0
        for i in range(len(x)):
            if(var[i] == 0):
                continue
            log_gauss = math.log(1/(2 * math.pi * var[i] **2)**0.5) - (x[i] - mu[i])**2 / (2 * var[i]**2)
            val = val + log_gauss
        return val
    
    def c_prob(self, _x, _c):
        prob = 1
        if _c is None:
            prob = self.gaussian(_x, self.mean, self.var)
        elif _c == 0:
            prob = self.prior_0 * self.gaussian(_x, self.var_0, self.mean_0)
        elif _c == 1:
            prob = self.prior_1 * self.gaussian(_x, self.var_1, self.mean_1)
        return prob
    
    def classify(self, data):
        predicted = []
        i = 0
        for line in data:
            i += 1
            wd = self.format_wordChoice()
            x = self.format_feature(wd, line)
            prob_0 = self.c_prob(x, 0)
            prob_1 = self.c_prob(x, 1)
            if  prob_0 >= prob_1:
                predicted.append(0)
            else:
                predicted.append(1)
            if(i % 100 == 0):
                print(i, 'in', len(data))
        #print('predicted', predicted)
        return predicted

wordListPurity = np.load('wordListPurity.npy')
#print(wordListPurity[0:10])

fpath = 'training.txt'

if not os.path.exists(fpath):
    print('file no exist')
f = open(fpath, encoding = 'utf-8')

data = f.readlines()

myChoice = []
j = 0
for i in range(int(wordListPurity.shape[0] * 0.1)):
    purity = float(wordListPurity[i,1])
    if(purity > 0.6 or purity < 0.4):
        j+=1
        myChoice.append(wordListPurity[i, 0])
        if(j < 50):
            print(wordListPurity[i])
print(len(wordListPurity), len(myChoice))

myclassifier = Nbclassifier(myChoice)
myWd = myclassifier.format_wordChoice()
linedata = data[0]
myFeature = myclassifier.format_feature(myWd, data[0])
print(linedata)
print(myclassifier.wordChoice[0:20])
print(myFeature)
print('training accuracy', myclassifier.train(data))
print('mean_0', myclassifier.mean_0)
print('prior_0', myclassifier.prior_0)
print('bad_count', myclassifier.bad_vector_count)
