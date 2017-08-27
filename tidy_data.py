from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import re
import pandas as pd 

class TidySymbols(BaseEstimator, TransformerMixin):
    
    emoji_pattern = re.compile("(["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002600-\U000026FF"  # misc sumbols
                               "]+)", flags=re.UNICODE)

    
    def __init__(self):
        pass
        
    def transform(self, X, *_):
        
        if type(X) == list:
            X = pd.Series(X)
        # Get rid of brackets
        X = X.str.replace(r"\(|\)|\[|\]"," ")
        
        X = X.str.replace(r"\\|\/|="," ")
        
        X = X.str.replace(r"\.\.\."," ")
        
        # Get rid of some other undesirable characters
        X = X.str.replace(r"\!|\+|\*|\/|\-|'|\\","")
        
        # X T C is a common thing
        X = X.str.replace("X T C", "XTC")
        X = X.str.replace("x t c", "XTC")
        
        # Sequences of emojis etc are factored out into their own symbol, space separated
        X = X.str.replace(self.emoji_pattern, " \\1 ")
        
        # 25x20 -> 25x 20
        p = re.compile("(\d+)(x|X)(\d+)")
        X = X.str.replace(p, "\\1x \\2 ")
        
        # 5 x -> 5x
        p = re.compile("(\d+) (x|X)")
        X = X.str.replace(p, "\\1x")
        
        
        p = re.compile("(\d+) mg")
        X.str.replace(p, "\\1mg")
        
        p = re.compile("(\d+) gr")
        X.str.replace(p, "\\1gr")
        
        p = re.compile("(\d+) g")
        X.str.replace(p, "\\1g")
        
        return X
        
    
    def fit(self, *_):
        return self