import sys, os
sys.path.append(".")
from Utils import evaluate
import pickle
import argparse

with open(r'data\tok_e.pickle', 'rb') as tok1:
    tok_eng = pickle.load(tok1)
with open(r'data\tok_h.pickle', 'rb') as tok2:
    tok_hin = pickle.load(tok2)

    

while True:
    s = input("Please enter any sentence in english")
    if s is None:
        print("Please enter any sentence in english or enter exit")
        exit()
    elif s == "exit":
        exit()

    else:
        result, sentence, attention_plot = evaluate.evaluate(s,tok_eng, tok_hin)

        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

    
