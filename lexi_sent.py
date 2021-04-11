#!/usr/bin/env python3
"""

"""
import nltk
import pandas as pd
import numpy as np
import argparse
import progressbar
import sys, os


def janis_fadner(pos, neg):
    """Returns Fanis-Fadner Coefficient of Imbalance"""
    jfci = [0]*len(pos)
    for i, (p, n) in enumerate(zip(pos, neg)):
        if p > n:
            jfci[i] = (p**2 - p * n) / (p + n)**2
        elif p==0 & n==0:
            jfci[i] = 0
        else:
            jfci[i] = (p * n - n**2) / (p + n)**2
    return jfci


def word_counter(words, text):
    """Vectorized string search"""
    total = [0]*len(text)
    for i, txt in enumerate(text):
        for word in words:
            if word in txt:
                total[i] = total[i] + 1
    return total


def sentiment_analysis(wordlist_file, csv_file):
    """Sentiment analysis routine"""
    
    # Get wordlist
    wordlist = pd.read_csv(wordlist_file)
    pos_words = wordlist[wordlist['sentiment'] > 0]['token'].to_list()
    neg_words = wordlist[wordlist['sentiment'] < 0]['token'].to_list()
    
    # Get input text
    df = pd.read_csv(csv_file, names=["Text"], encoding='latin1')
    
    # Calculate sentiment
    df['pos_words'] = word_counter(pos_words, df['Text'])
    df['neg_words'] = word_counter(neg_words, df['Text'])
    df['Sentiment'] = janis_fadner(df['pos_words'], df['neg_words'])
    
    df = df[['Text', 'Sentiment']]
    
    return df
    

def main():
    # Get command line arguments 
    parser = argparse.ArgumentParser(description='Perform sentiment analysis on a list of documents.')
    parser.add_argument('input', type=str, nargs=1, help='A CSV file with a single column, containing the text to one document per row') #
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-w','--wordlist', help='CSV file containing a word list with positive and negative words. Default is the MPQA word list, which ships with this script. Different files must follow the same format.', required=False, nargs=1) #
    group.add_argument('-o','--output', help='Name for output file. Defaults to "Sentiments.csv"', required=False, nargs=1) #
   
    # Parse arguments
    args = vars(parser.parse_args())
    csv_file = args['input'][0]
    if args['wordlist'] is not None:
        wordlist_file = args['wordlist'][0]
    else:
        wordlist_file = "MPQA.csv"
    if args['output'] is not None:
        output_file = args['output'][0]
    else:
        output_file = 'Sentiments.csv'
        
    # Sentiment analysis
    results = sentiment_analysis(wordlist_file, csv_file)
    
    # Export results
    results.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
