import numpy as np
import os
import matplotlib.pyplot as plt


# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

sortedWordList = np.load('sortedWordList.npy')
word_count = np.load('wordCount.npy')


sortedWordBag = dict(sortedWordList)
print(sortedWordList.shape)
print(sortedWordList[0:5])
print(sortedWordList.dtype)
nil_arr = []
one_arr = []
labels = []
wordListPurity = []
for key in sortedWordBag:
    nil_pct = (sortedWordBag[key][0]-sortedWordBag[key][1])/word_count[0]
    one_pct = sortedWordBag[key][1]/word_count[1]
    nil_portion = nil_pct/(one_pct + nil_pct)
    labels.append(key + " {:.3f}".format(nil_portion))
    wordListPurity.append((key , nil_portion))
    nil_arr.append(nil_pct)
    one_arr.append(one_pct)
wordListPurity = np.array(wordListPurity)
print(wordListPurity.shape)
print(wordListPurity[0:5])
print(wordListPurity.dtype)
#np.save('wordListPurity.npy', wordListPurity)

for hcent in range(10):
    fig, ax = plt.subplots()
    width = 0.35
    plt.xticks(rotation=90)
    ax.bar(labels[hcent*50:hcent*50 + 50], nil_arr[hcent*50:hcent*50 + 50], width, bottom= one_arr[hcent*50:hcent*50 + 50], label='negative, 0')
    ax.bar(labels[hcent*50:hcent*50 + 50], one_arr[hcent*50:hcent*50 + 50], width, label='positive, 1')
    ax.set_title('entry ' + str(hcent * 50) + ' to ' + str(hcent * 50 + 50) + ' in ' + str(10*50))
    ax.legend()
    plt.show()
    print('neg review words', word_count[0])
    print('pos review words', word_count[1])
    print('nilsum', np.sum(nil_arr))
    print('onesum', np.sum(one_arr))
    print(nil_arr[0])
    print(one_arr[0])
    print('total number of unique words', len(sortedWordList))

