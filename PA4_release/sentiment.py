"""
CS311 Programming Assignment 4: Naive Bayes

Full Name: AJ Noyes

Brief description of my custom classifier: My classifier uses unigrams, ngrams, document length, and capitalization ratio as features. I also implemented weights for each feature, and I 
modified the pseudo count and ngram length until performance was improved. 
"""
import argparse, math, os, re, string, zipfile
from typing import Generator, Hashable, Iterable, List, Sequence, Tuple
import numpy as np
from sklearn import metrics


class Sentiment:
    """Naive Bayes model for predicting text sentiment"""

    def __init__(self, labels: Iterable[Hashable]):
        """Create a new sentiment model

        Args:
            labels (Iterable[Hashable]): Iterable of potential labels in sorted order.

        """
        # Vocab
        self.vocabulary = set()
        # Labels
        self.labels = list(labels)
        # Word count dictionaries by label
        self.word_counts = {label: {} for label in self.labels}
        # Total document counts for each label
        self.doc_counts = {label: 0 for label in self.labels}
        # Total word counts for each label
        self.total_word_counts = {label: 0 for label in self.labels}

    def preprocess(self, example: str, id:str =None) -> List[str]:
        """Normalize the string into a list of words.

        Args:
            example (str): Text input to split and normalize
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            List[str]: Normalized words
        """
        # TODO: Modify the method to generate individual words from the example. Example modifications include
        # removing punctuation and/or normalizing case (e.g., making all lower case)

        example = example.lower()
        
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        cleaned_text = example.translate(translator)
        
        # Split into words
        words = cleaned_text.split()
        
        return words

    def add_example(self, example: str, label: Hashable, id:str = None):
        """Add a single training example with label to the model

        Args:
            example (str): Text input
            label (Hashable): Example label
            id (str, optional): File name from training/test data (may not be available). Defaults to None.
        """
        # TODO: Implement function to update the model with words identified in this training example

        processed = self.preprocess(example, id)

        self.doc_counts[label] += 1

        # Update counts
        for word in processed:
            # Add to vocab
            self.vocabulary.add(word)
            
            # Update word by label & word
            if word not in self.word_counts[label]:
                self.word_counts[label][word] = 1
            else:
                self.word_counts[label][word] += 1
            
            # Update total word count for this label
            self.total_word_counts[label] += 1

    def predict(self, example: str, pseudo=0.0001, id:str = None) -> Sequence[float]:
        """Predict the P(label|example) for example text, return probabilities as a sequence

        Args:
            example (str): Test input
            pseudo (float, optional): Pseudo-count for Laplace smoothing. Defaults to 0.0001.
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            Sequence[float]: Probabilities in order of originally provided labels
        """
        # Preprocess the text
        words = self.preprocess(example, id)
        
        # Total number of documents
        total_docs = sum(self.doc_counts.values())
        
        # Log probabilities for each label
        log_probs = []
        
        for label in self.labels:
            # Prior probability (log space)
            prior_prob = np.log(self.doc_counts[label] / total_docs)
            
            # Compute log likelihood for each word
            log_likelihood = prior_prob
            for word in words:
                # Laplace smoothing
                word_count = self.word_counts[label].get(word, 0)
                total_word_count = self.total_word_counts[label]
                
                # Log probability of word given label

                word_prob = np.log(
                    (word_count + pseudo) / 
                    (total_word_count + pseudo * (len(self.word_counts[label]))
                ))
                
                log_likelihood += word_prob
            
            log_probs.append(log_likelihood)
        
        # Convert log probabilities to probabilities using softmax
        probs = np.exp(log_probs)
        probs /= np.sum(probs)
        
        return probs.tolist()

class CustomSentiment(Sentiment):
    # TODO: Implement your custom Naive Bayes model
    def __init__(self, labels: Iterable[Hashable], ngram_length: int = 4):
        """
        Initialize the custom sentiment classifier

        Args:
            labels (Iterable[Hashable]): Possible labels
            ngram_length (int, optional): Length of character n-grams. Defaults to 4.
        """
        super().__init__(labels)
        
        # Character n-gram specific attributes
        self.ngram_length = ngram_length
        self.ngram_vocab = set()
        self.ngram_word_counts = {label: {} for label in self.labels}
        self.total_ngram_word_counts = {label: 0 for label in self.labels}

    def generate_ngrams(self, word: str) -> List[str]:
        """Generate character n-grams for a given word."""
        if len(word) < self.ngram_length:
            return []
        return [word[i:i+self.ngram_length] for i in range(len(word)-self.ngram_length+1)]

    def preprocess(self, example: str, id:str =None) -> List[str]:
        """Enhanced preprocessing with character n-grams, document length, and capitalization ratio."""
        # Base preprocessing (lowercase, remove punctuation)
        example = example.lower()
        translator = str.maketrans('', '', string.punctuation)
        cleaned_text = example.translate(translator)
        
        # Words
        words = cleaned_text.split()
        
        # Compute document length and capitalization features
        total_chars = len(example)
        document_length = len(words)
        capitalization_ratio = len(re.findall(r'[A-Z]', example)) / total_chars if total_chars > 0 else 0
        
        # Generate n-grams 
        char_ngrams = []
        for word in words:
            char_ngrams.extend(self.generate_ngrams(word))
        
        # Doc Length and Cap Ratio combined
        enhanced_features = words + char_ngrams + [
            f"DOC_LENGTH_{document_length}",
            f"CAP_RATIO_{capitalization_ratio:.4f}"
        ]
        
        return enhanced_features

    def add_example(self, example: str, label: Hashable, id:str = None):
        """Enhanced example addition with character n-grams, document length, and capitalization."""
        processed = self.preprocess(example, id)

        self.doc_counts[label] += 1

        # Track feature additions
        for word in processed:
            if word.startswith(("DOC_LENGTH_", "CAP_RATIO_")):
                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 1
                else:
                    self.word_counts[label][word] += 1
                self.total_word_counts[label] += 1
                self.vocabulary.add(word)
            
            # Character n-grams
            elif len(word) == self.ngram_length and word.isalpha():
                self.ngram_vocab.add(word)
                if word not in self.ngram_word_counts[label]:
                    self.ngram_word_counts[label][word] = 1
                else:
                    self.ngram_word_counts[label][word] += 1
                self.total_ngram_word_counts[label] += 1
            
            else:
                self.vocabulary.add(word)
                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 1
                else:
                    self.word_counts[label][word] += 1
                self.total_word_counts[label] += 1

    def predict(self, example: str, pseudo=1, id:str = None) -> Sequence[float]:
        """Enhanced prediction with character n-grams, document length, and capitalization."""
        # Preprocess with multiple features
        words = self.preprocess(example, id)
        
        # Total number of documents
        total_docs = sum(self.doc_counts.values())
        
        # Log probabilities for each label
        log_probs = []

        # Weights 
        data_weight = 1.5 # 1.5
        n_weight = 0.25 #0.25
        reg_weight = 0.25 #0.25
        
        for label in self.labels:
            # Priors
            prior_prob = np.log(self.doc_counts[label] / total_docs)
            
            # Compute log likelihood
            log_likelihood = prior_prob
            
            for word in words:
                # Data Features
                if word.startswith(("DOC_LENGTH_", "CAP_RATIO_")):
                    word_count = self.word_counts[label].get(word, 0)
                    total_word_count = self.total_word_counts[label]
                    
                    word_prob = np.log(
                        (word_count + pseudo) / 
                        (total_word_count + pseudo * (len(self.word_counts[label])))
                    )
                    log_likelihood += (word_prob * data_weight)
                
                # Character n-grams
                elif len(word) == self.ngram_length and word.isalpha():
                    word_count = self.ngram_word_counts[label].get(word, 0)
                    total_word_count = self.total_ngram_word_counts[label]
                    
                    word_prob = np.log(
                        (word_count + pseudo) / 
                        (total_word_count + pseudo * (len(self.ngram_vocab)))
                    )
                    log_likelihood += (word_prob * n_weight)
                
                # Regular words
                else:
                    word_count = self.word_counts[label].get(word, 0)
                    total_word_count = self.total_word_counts[label]
                    
                    word_prob = np.log(
                        (word_count + pseudo) / 
                        (total_word_count + pseudo * (len(self.vocabulary)))
                    )
                    log_likelihood += (word_prob * reg_weight)
            
            log_probs.append(log_likelihood)
        
        # Convert log probabilities
        probs = np.exp(log_probs)
        probs /= np.sum(probs)
        
        return probs.tolist()

def process_zipfile(filename: str) -> Generator[Tuple[str, str, int], None, None]:
    """Create generator of labeled examples from a Zip file that yields a tuple with
    the id (filename of input), text snippet and label (0 or 1 for negative and positive respectively).

    You can use the generator as a loop sequence, e.g.

    for id, example, label in process_zipfile("test.zip"):
        # Do something with example and label

    Args:
        filename (str): Name of zip file to extract examples from

    Yields:
        Generator[Tuple[str, str, int], None, None]: Tuple of (id, example, label)
    """
    with zipfile.ZipFile(filename) as zip:
        for info in zip.infolist():
            # Iterate through all file entries in the zip file, picking out just those with specific ratings
            match = re.fullmatch(r"[^-]+-(\d)-\d+.txt", os.path.basename(info.filename))
            if not match or (match[1] != "1" and match[1] != "5"):
                # Ignore all but 1 or 5 ratings, 5 = LEFT, 1 = RIGHT
                continue
            # Extract just the relevant file the Zip archive and yield a tuple
            with zip.open(info.filename) as file:
                yield (
                    match[0],
                    file.read().decode("utf-8", "ignore"),
                    1 if match[1] == "5" else 0,
                )


def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Naive Bayes sentiment analyzer")

    parser.add_argument(
        "--train",
        default="data/train.zip",
        help="Path to zip file or directory containing training files.",
    )
    parser.add_argument(
        "--test",
        default="data/test.zip",
        help="Path to zip file or directory containing testing files.",
    )
    parser.add_argument(
        "-m", "--model", default="base", help="Model to use: One of base or custom"
    )
    parser.add_argument("example", nargs="?", default=None)

    args = parser.parse_args()

    # Train model
    if args.model == "custom":
        model = CustomSentiment(labels=[0, 1])
    else:
        model = Sentiment(labels=[0, 1])
    for id, example, y_true in process_zipfile(
        os.path.join(os.path.dirname(__file__), args.train)
    ):
        model.add_example(example, y_true, id=id)

    # If interactive example provided, compute sentiment for that example
    if args.example:
        print(model.predict(args.example))
    else:
        predictions = []
        for id, example, y_true in process_zipfile(
            os.path.join(os.path.dirname(__file__), args.test)
        ):
            # Determine the most likely class from predicted probabilities
            predictions.append((id, y_true, np.argmax(model.predict(example,id=id))))

        # Compute and print accuracy metrics
        _, y_test, y_true = zip(*predictions)
        predict_metrics = compute_metrics(y_test, y_true)
        for met, val in predict_metrics.items():
            print(
                f"{met.capitalize()}: ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )
