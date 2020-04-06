import numpy as np
from functools import reduce
import math
import warnings
from sklearn.ensemble import IsolationForest
warnings.filterwarnings("ignore")


class GroupColumn:
    
    def __init__(self, columns, value_mapping):
        self.columns = columns    
        self.value_mapping = value_mapping
        self.col_values = set()
        
    def convert(self, df):
        x = df.copy()
        for c in self.columns:
            x[c] = x[c].map(self.value_mapping)
        return x[self.columns]
        
    def get_mapping(self, df):
        x = self.convert(df)
        maps = {}
        all_values = set(np.unique(x))
        for c in self.columns:
            freq = x[c].value_counts(normalize=True).to_dict()
            self.col_values.union(set(freq.keys()))
            missing = all_values - set(freq.keys())
            for key in missing:
                freq[key] = 0
            maps[c] = freq

       # self.freq_map = maps


    @staticmethod
    def seq_comb(sequence):
        sequence = list(map(int, sequence))
        count = {k: sequence.count(k) for k in set(sequence)}
        num = math.factorial(len(sequence))
        den =  reduce(lambda x,y: x * y, [math.factorial(v) for k,v in count.items()])
        return math.log(num / den)

    @staticmethod
    def seq_prob(sequence, values, prob):
        count = {k: sequence.count(k) for k in set(values)}
        return sum([np.log(p) * count.get(values[i], 0) for i, p in enumerate(prob)])
    
    def score(self, df, incude_prob=False, percentile=None):
        x = self.convert(df)
        x['score'] = 1
        for k in gc.columns:
            x['score'] *= x[k].map(self.freq_map[k])  
        combinatorics = np.log(x['score']) 
        
        if incude_prob:
            score = combinatorics + x.apply(GroupColumn.seq_comb, axis=1)
        else:
            score = combinatorics - len(self.value_mapping) ** len(self.columns)
            
        if percentile is not None:
            return (score <= np.percentile(score, percentile)) * 1
        else:
            return score
        
    def isolation_forest(self, df, percentile=None):
        x = self.convert(df)
        ifs = IsolationForest(bootstrap=True, max_features=0.75)
        ifs.fit(x)
        score = ifs.score_samples(x)
        if percentile is not None:
            return (score >= np.percentile(score, percentile)) * 1
        else:
            return score


class Mappings:

    likert = {
        'Very likely': 2,
        'Somewhat likely': 1,
        'Somewhat unlikely': -1,
        'Very unlikely': -2,
    }

    true_false = {
        'True': 1,
        'False': -1,
        'Not sure': 0
    }

    app = {
        'Stayed the same': 0,
        'Doing a lot more': 2,
        'Doing slightly more': 1,
        'Doing slightly less': -1,
        'Doing a lot less': -2,
        'Does not apply to me': 99
    }

    how_often = {
        'Never': 0,
        'Once': 1,
        'Sometimes': 2,
        'Many times': 3,
        'Constantly': 4
    }

    good_in_me = {
        'Mostly bad in me': -2.0,
        'Only the worst in me': -1.0,
        'Equally good and bad in me': 0.0,
        'Mostly the good in me': 1.0,
        'Only the best in me': 2.0,
    }

    productivity = {
        'Doing a lot more': 2.0,
        'Doing slightly more': 1.0,
        'Stayed the same': 0.0,
        'Doing slightly less': -1.0,
        'Doing a lot less': -2.0,
    }


