# NB Political Sentiment Analyzer
Project for CS311. Naive Bayes model. 

All training and testing data have been omitted from this repository for display purposes. All data sourced from Kaggle. 
Displayed around 85 percent accuracy as coded. All training or testing data should be formatted as:

[name]-[value]-[#}.txt

Value should either be 1 or 5, 1 representing liberal leaning, and 5 representing conservative leaning. Liberal leaning corresponds with the 0 (negative) label, and conservative corresponds to the 1 label, and the results should be interpreted as such. # can be any digit, and name can be any text.

Features included on this model:
- Naive Bayes classification
- Phrase weighting
- Negation handling
- POS Tagging
- n-gram support
- Evaluation metrics
- Zip file processing

Before implementing, you must run:

pip install numpy scikit-learn nltk
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
