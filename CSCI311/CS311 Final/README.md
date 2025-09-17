# CS311Final
Final for CS311. Naive Bayes political sentiment analyzer. 

All training and testing data have been omitted from this repository for display purposes. All data sourced from Kaggle. 
Displayed around 85 percent accuracy as coded.

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
