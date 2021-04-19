#!/usr/bin/env python3
"""

"""
import pandas as pd
import numpy as np
import argparse
import os
import re # Regex
import nltk
import io # Handles encoding of text files

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
    total = [0]*len(text) # Empty list
    for i, txt in enumerate(text):
        for word in words:
            if word in txt:
                total[i] = total[i] + 1
    return total


def sentiment_analysis(df, wordlist):
    """Sentiment analysis routine using janis fadner coefficient of imbalance"""

    # Get wordlist
    pos_words = wordlist[wordlist['sentiment'] > 0]['token'].to_list()
    neg_words = wordlist[wordlist['sentiment'] < 0]['token'].to_list()
        
    # Calculate sentiment
    df['PositiveWords'] = word_counter(pos_words, df['Text'])
    df['NegativeWords'] = word_counter(neg_words, df['Text'])
    df['Sentiment'] = janis_fadner(df['PositiveWords'], df['NegativeWords'])
    
    return df


def clean_doc(doc):
    """Cleans a document, extracts meta data, returns a dictionary"""
    
    # Split header and text
    topmarker = "Body"
    if re.search("\n" + topmarker + ".?\n", doc) is not None:
        headersplit = re.split("\n" + topmarker + ".?\n", doc)
        header = headersplit[0]
        body = headersplit[1]
        cleaned = 1
    else:
        body = doc
        header = ''
        cleaned = 0
    
    # Try getting the date
    try:
        dateresult = re.findall(r'\n\s{5}.*\d+.*\d{4}\s', header, flags=re.IGNORECASE)
        if header:
            dateresult += re.findall(r'\w+\s\d+.*\d{4}', header)
            dateresult += re.findall(r'\w+\s*\d{4}', header)
        date = dateresult[0].strip()
    except:
        date = ''
    
    # Clean text body
    words = nltk.word_tokenize(body) # Tokenize words
    words = [w.lower() for w in words] # Lowercase everything
    words = list(set(words)) # Unique words only
    words = [w for w in words if w.isalpha()] # Letters only
    text = ' '.join(words)
    
    # Collect results
    cleaned_doc = { 
        'Text': text,
        'Date': date
    }
    
    return cleaned_doc


def folder_import(path):
    """Function imports each document in path, cleans it, and appends to a data frame"""
    files = os.listdir(path)
    # Text files only
    files = [f for f in files if f.split(".")[-1]=="txt"]
    # Results table
    df = pd.DataFrame()
    # Loop through files in folder
    for i, f in enumerate(files):
        # Read file
        fp = io.open(os.path.join(path, f), 'r', encoding='windows-1252').read()
        # Clean file
        fp_clean = clean_doc(fp)
        # Add file name to results
        fp_clean['File'] = f
        # Append results to dataframe
        df = df.append(fp_clean, ignore_index=True)
    return df
    

def main():
    # Get command line arguments 
    parser = argparse.ArgumentParser(description='Perform sentiment analysis on a list of documents.')
    parser.add_argument('input', type=str, nargs=1, help='A CSV file with a single column, containing the text to one document per row') #
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-w','--wordlist', help='CSV file containing a word list with positive and negative words. Default is the MPQA word list, which ships with this script. Different files must follow the same format.', required=False, nargs=1, default='MPQA.csv') #
    group.add_argument('-o','--output', help='Name for output file. Defaults to "Sentiments.csv"', required=False, nargs=1, default='Sentiments.csv') #
   
    # Parse arguments
    args = vars(parser.parse_args())
    input_arg = args['input'][0]
    if args['wordlist'] is not None:
        wordlist_file = args['wordlist']
    if args['output'] is not None:
        output_file = args['output']
    
    # Import text data
    if input_arg.split(".")[-1]=="csv": # Check if input is csv file
        text_data = pd.read_csv(input_arg, names=["Text"], encoding='latin1')
    elif os.path.isdir(input_arg): # Check if input is a folder
        text_data = folder_import(input_arg)
    else:
        raise ValueError("input should be path to a folder or csv file")
        
    # Import wordlist
    wordlist = pd.read_csv(wordlist_file)
    
    # Sentiment analysis
    results = sentiment_analysis(text_data, wordlist)
    
    # Export results
    results.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
