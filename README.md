# lexi_sent

A python3 script for lexicon-based sentiment analysis.

## Usage

The program runs from the command line. Input can be a csv file or a folder. 

To see the program help:

	python3 lexi_sent.py --help 

For CSV import, run the following commands:

	python3 lexi_sent.py test.csv 

The input should be a CSV file with a single unnamed column, containing a document per row (see test.csv). 

The folder import expects the input to be a folder. Only `txt` files can be imported. Example call:

	python3 lexi_sent.py test
		
Optional arguments: 

 - `-o` to specify a custom output file (defaults to 'Sentiments.csv')
 - `-w` to specify a custom wordlist. Defaults to 'MPQA.csv'. Custom wordlists need to be in the same format as the default wordlist. 
 - `-v` prints name of each file during folder import, for tracing of encoding or other issues. 


## FAQ

### How to install dependencies?

If you're getting the error "ImportError: No module named ...", install the missing package(s):

	pip3 install pandas numpy openpyxl
		
### How to install Python?

If you're on Mac or Linux, Python is already installed. On windows, get [the latest version from the Python website](https://www.python.org/downloads/windows/)

## Citation

For usage in any published material, please cite:

> Nauhaus, S. (2020). The Role of Stakeholder Sentiment in Strategic Decision-Making: A Behavioral Perspective. Geneva: University of Geneva.

If you're using the default [MPQA dictionary](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/), the authors would appreciate a citation as well:

> Theresa Wilson, Janyce Wiebe, & Paul Hoffmann (2005). Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis. Proc. of HLT-EMNLP-2005.
