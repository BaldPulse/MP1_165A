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
        
