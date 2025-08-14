import zipfile, argparse, os
import nltk
from typing import Generator, Hashable, Iterable, List, Sequence, Tuple
from sklearn import metrics
import numpy as np
import re
import string

# Requires running nltk.download('punkt') and nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

class Sentiment:
    """Naive Bayes model for predicting text sentiment (e.g., political views)"""

    def __init__(self, labels: Iterable[Hashable]):
        self.vocabulary = set()
        self.labels = list(labels)
        self.word_counts = {label: {} for label in self.labels}
        self.doc_counts = {label: 0 for label in self.labels}
        self.total_word_counts = {label: 0 for label in self.labels}
        # Add phrase weighting
        self.phrase_weights = {
        
        # Conservative
        "tax cuts": 2,
        "law and order": 1.5,
        "family values": 2,
        "strong military": 1.5,
        "fossil fuels": 2,
        "family values": 2,
        "pro-life": 2,
        "MAGA" : 2,
        "Trump" : 1.5,
        "radical left": 2,
        "liberalists": 2,
        # Liberal
        "minimum wage": 2,
        "diversity": 2,
        "inequality": 2,
        "social justice": 2,
        "universal healthcare": 2,
        "climate crisis": 2,
        "gun control": 1.5,
        "minimum wage increase": 1.5,
        "black lives matter": 2,
        }

    def preprocess(self, example: str, id: str = None, n: int = 1) -> List[str]:
        """Normalize the string into a list of n-grams, handling phrases and negation.

        Args:
            example (str): Text input to split and normalize.
            id (str, optional): File name from training/test data (may not be available). Defaults to None.
            n (int, optional): Size of the n-grams. Defaults to 1.

        Returns:
            List[str]: Normalized words or n-grams.
        """
        example = example.lower()
        translator = str.maketrans('', '', string.punctuation)
        cleaned_text = example.translate(translator)
        words = cleaned_text.split()

        # Perform NLTK Part of Speech Tagging
        pos_tagged_words = nltk.pos_tag(words)
        
        # Custom dictionary for specific terms (keeping the original bias-related terms)
        custom_pos_tags = {
            # Political Bias Terms (Liberal leaning)
            'progressive': 'JJ',
            'liberal': 'JJ',
            'democratic': 'JJ',
            'socialist': 'JJ',
            'inclusive': 'JJ',
            'equality': 'NN',
            'diversity': 'NN',
            'civil rights': 'NNS',
            'immigrant': 'NN',
            'climate change': 'NN',
            'green new deal': 'NN',
            'gun control': 'NN',
            'abortion rights': 'NN',
            'human rights': 'NN',
            'inequality': 'NN',
            'social justice': 'NN',
            # Political Bias Terms (Conservative leaning)
            'conservative': 'JJ',
            'right-wing': 'JJ',
            'traditional': 'JJ',
            'capitalism': 'NN',
            'freedom': 'NN',
            'patriotism': 'NN',
            'family values': 'NNS',
            'pro-life': 'JJ',
            'second amendment': 'NN',
            'small government': 'NN',
            'law and order': 'NN',
            'self-reliance': 'NN',
            'MAGA': 'NN',
            'radical': 'JJ',
        }
        
        # Tag words with either NLTK's POS tag or custom tag
        tagged_words = []
        for word, pos in pos_tagged_words:
            # Check if the word is in our custom dictionary first
            if word in custom_pos_tags:
                tagged_words.append(f"{word}_{custom_pos_tags[word]}")
            else:
                # Use NLTK's POS tag, with a fallback to 'NN'
                # Map Penn Treebank tags to simpler tags
                simple_pos = {
                    'JJ': 'JJ',   # Adjective
                    'JJR': 'JJ',  # Adjective, comparative
                    'JJS': 'JJ',  # Adjective, superlative
                    'NN': 'NN',   # Noun, singular
                    'NNS': 'NN',  # Noun, plural
                    'NNP': 'NN',  # Proper noun, singular
                    'NNPS': 'NN', # Proper noun, plural
                    'VB': 'VB',   # Verb, base form
                    'VBD': 'VB',  # Verb, past tense
                    'VBG': 'VB',  # Verb, gerund/present participle
                    'VBN': 'VB',  # Verb, past participle
                    'VBP': 'VB',  # Verb, non-3rd person singular present
                    'VBZ': 'VB',  # Verb, 3rd person singular present
                }.get(pos, 'NN')  # Default to 'NN' if no match
                
                tagged_words.append(f"{word}_{simple_pos}")


        # Negation handling
        negations = {"not", "never", "no", "cannot", "doesn't", "isn't", "wasn't", "won't", "don't", "didn't", "aren't", "can't"}
        negated_words = []
        negate = False
        for word in words:
            if word in negations:
                negate = True
            elif negate:
                negated_words.append(f"{word}_NEG")
                if len(negated_words) >= 3:  # Stop negating after 3 words
                    negate = False
            else:
                negated_words.append(word)

        # Apply phrase weighting
        for phrase, weight in self.phrase_weights.items():
            if phrase in cleaned_text:
                negated_words.append(phrase)  # Treat the phrase as a single token
                for _ in range(int(weight - 1)):  # Increase its count based on weight
                    negated_words.append(phrase)

        # Generate n-grams
        if n > 1:
            ngrams = [' '.join(negated_words[i:i + n]) for i in range(len(negated_words) - n + 1)]
            pos_ngrams = [' '.join(tagged_words[i:i + n]) for i in range(len(tagged_words) - n + 1)]
            return ngrams + pos_ngrams

        return negated_words + tagged_words

    def add_example(self, example: str, label: Hashable, id: str = None, n: int = 1):
        processed = self.preprocess(example, id, n)
        self.doc_counts[label] += 1
        for word in processed:
            self.vocabulary.add(word)
            if word not in self.word_counts[label]:
                self.word_counts[label][word] = 1
            else:
                self.word_counts[label][word] += 1
            self.total_word_counts[label] += 1

    def predict(self, example: str, pseudo=1.0, id: str = None, n: int = 1) -> Sequence[float]:
        words = self.preprocess(example, id, n)
        total_docs = sum(self.doc_counts.values())
        log_probs = []

        for label in self.labels:
            prior_prob = np.log(self.doc_counts[label] / total_docs)
            class_weight = total_docs / (self.doc_counts[label] + 1)
            prior_prob += np.log(class_weight)
            vocab_size = len(self.vocabulary)
            log_likelihood = prior_prob

            for word in words:
                word_count = self.word_counts[label].get(word, 0)
                total_word_count = self.total_word_counts[label]
                word_prob = np.log(
                    (word_count + pseudo) /
                    (total_word_count + pseudo * (vocab_size + 1))
                )
                log_likelihood += word_prob

            log_probs.append(log_likelihood)

        # Convert log probabilities to probabilities using a stable softmax
        max_log_prob = max(log_probs)
        log_probs = [log_prob - max_log_prob for log_prob in log_probs]
        probs = np.exp(log_probs)
        probs /= np.sum(probs)

        return probs.tolist()


def process_zipfile(filename: str) -> Generator[Tuple[str, str, int], None, None]:
    with zipfile.ZipFile(filename) as zip:
        for info in zip.infolist():
            match = re.fullmatch(r"[^_]+_(\d)_\d+\.txt", os.path.basename(info.filename))
            if not match or (match[1] != "1" and match[1] != "5"):
                continue
            label = 1 if match[1] == "5" else 0
            with zip.open(info.filename) as file:
                yield (
                    info.filename,
                    file.read().decode("utf-8", "ignore"),
                    label,
                )


def compute_metrics(y_true, y_pred):
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
        "--ngrams",
        type=int,
        default=1,
        help="Size of n-grams to use for preprocessing (default: 1 for unigrams).",
    )
    parser.add_argument("example", nargs="?", default=None)
    args = parser.parse_args()

    print("Starting training...")
    model = Sentiment(labels=[0, 1])
    for id, example, y_true in process_zipfile(
        os.path.join(os.path.dirname(__file__), args.train)
    ):
        model.add_example(example, y_true, id=id, n=args.ngrams)

    if args.example:
        print(model.predict(args.example, n=args.ngrams))
    else:
        predictions = []
        for id, example, y_true in process_zipfile(
            os.path.join(os.path.dirname(__file__), args.test)
        ):
            predictions.append((id, y_true, np.argmax(model.predict(example, id=id, n=args.ngrams))))

        _, y_test, y_true = zip(*predictions)
        predict_metrics = compute_metrics(y_test, y_true)
        for met, val in predict_metrics.items():
            print(
                f"{met.capitalize()}: ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )
