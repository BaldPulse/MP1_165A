import numpy as np
import os
import matplotlib.pyplot as plt
import operator

fpath = 'training.txt'

if not os.path.exists(fpath):
    print('file no exist')
f = open(fpath, encoding = 'utf-8')
print(f)
lines = f.readlines()

wordBag = {}
i = 0
nil_count = 0
one_count = 0
for line in lines:
    word = ''
    label = int(line[-2])
    #if(label == 1):
    #    one_count += 1
    #else:
    #    nil_count += 1 
    line_dict = {}
    for c in line:
        if c == ' ':
            if wordBag.get(word) is None:
                wordBag.update({word : [1, label]})
            else:
            #elif line_dict.get(word) is None:
            #    line_dict.update({word:1})
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
    i = i + 1
    if i % 1000 == 0:
        print(i, label) 
    #if i > 100:
    #    break

sortedWordList = np.array(sorted(wordBag.items(), key=lambda item: item[1][0], reverse=True))
print('type sortedWordList', type(sortedWordList))
np.save('sortedWordList.npy', sortedWordList)
word_count = np.array([nil_count, one_count])
np.save('wordCount.npy', word_count)
sortedWordBag = dict(sortedWordList[0:50])
nil_arr = []
one_arr = []
labels = []
#print(sortedWordBag)
print(word_count)
for key in sortedWordBag:
    labels.append(key)
    nil_arr.append((sortedWordBag[key][0]-sortedWordBag[key][1])/word_count[0])
    one_arr.append(sortedWordBag[key][1]/word_count[1])


fig, ax = plt.subplots()
width = 0.35
ax.bar(labels, nil_arr, width,  label='negative')
ax.bar(labels, one_arr, width, bottom= nil_arr,  label='positive')
print('neg review words', word_count[0])
print('pos review words', word_count[1])
print('nilsum', np.sum(nil_arr))
print('onesum', np.sum(one_arr))
ax.legend()
plt.show()
