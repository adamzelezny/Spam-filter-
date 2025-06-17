import os
import math
from collections import Counter
  
"""Tento spam filter využívá Bayesův teorém, rozšírený o Laplaceovu úpravu k rozlišení mezi SPAM a HAM. Zároveň bere v potaz i adresy odesílatelů emailů."""
  
class MyFilter: #inicialization of the class MyFilter
    def __init__(self, laplace=1):
        self.spam_adresses = []
        self.laplace = laplace
        self.spam_word_probs = {}
        self.ham_word_probs = {}
        self.spam_prior = 0
        self.ham_prior = 0
        self.trained = False
  
    def preprocess(self, text): #preprocessing the input text by converting into lowercase letters and splitting into words
        return "".join(c.lower() if c.isalnum() else " " for c in text).split()
  
    def getadress(self, file_lines): #extracting the email address from "From:"
        for line in file_lines:
            if "From:" in line:
                for word in line.split():
                    if "@" in word:
                        return word
        return None
  
    def train(self, train_corpus_dir):
        truth_file = os.path.join(train_corpus_dir, "!truth.txt") #training the spamfilter 
        if not os.path.exists(truth_file):
            print("Corpus missing.")
            return
  
        with open(truth_file, "r", encoding="utf-8") as f:
            truth_data = [line.strip().split() for line in f]
  
        spam_files = [os.path.join(train_corpus_dir, fname) for fname, label in truth_data if label == "SPAM"] #splitting the given files into SPAM or HAM
        ham_files = [os.path.join(train_corpus_dir, fname) for fname, label in truth_data if label == "OK"]
  
        self.spam_prior = len(spam_files) / len(truth_data) #calculating priors 
        self.ham_prior = len(ham_files) / len(truth_data)
  
        spam_words = []
        ham_words = []
  
        for file_path in spam_files: #taking words and addresses from the spam files
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                spam_words.extend(self.preprocess(content))
                address = self.getadress(content.splitlines())
                if address:
                    self.spam_adresses.append(address)
  
        for file_path in ham_files: #taking words and addresses from the ham files
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                ham_words.extend(self.preprocess(content))
  
        spam_word_freq = Counter(spam_words) #calculating the frequencies of different words
        ham_word_freq = Counter(ham_words)
  
        vocab = set(spam_word_freq.keys()).union(ham_word_freq.keys())
        vocab_size = len(vocab)
  
        self.spam_word_probs = {
            word: (spam_word_freq[word] + self.laplace) / (len(spam_words) + self.laplace * vocab_size)
            for word in vocab
        }
  
        self.ham_word_probs = {
            word: (ham_word_freq[word] + self.laplace) / (len(ham_words) + self.laplace * vocab_size)
            for word in vocab
        }
  
        self.trained = True
  
    def logarithm(self, x): #logarithms are used to avoid rounding down to 0 for messages of very small probabilities 
        return math.log(x) if x > 0 else float('-inf')
  
    def predict(self, message): #predicting whether a message is SPAM or HAM
        words = self.preprocess(message)
        spam_score = self.logarithm(self.spam_prior)
        ham_score = self.logarithm(self.ham_prior)
  
        for word in words: #adding logarithm probablities of words for SPAM or HAM
            spam_score += self.logarithm(self.spam_word_probs.get(word, self.laplace / (len(self.spam_word_probs) + 1)))
            ham_score += self.logarithm(self.ham_word_probs.get(word, self.laplace / (len(self.ham_word_probs) + 1)))
        return "SPAM" if spam_score > ham_score else "OK"
  
    def test(self, test_corpus_dir): #testing the filter on a group of emails and coming up with a prediction
        prediction_file = os.path.join(test_corpus_dir, "!prediction.txt")
  
        with open(prediction_file, "w", encoding="utf-8") as pred_file:
            for file_name in os.listdir(test_corpus_dir):
                if file_name.startswith("!"):
                    continue
  
                file_path = os.path.join(test_corpus_dir, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    file_lines = content.splitlines()
                    address = self.getadress(file_lines)
                    prediction = self.predict(content)
                    if address in self.spam_adresses:
                        prediction = "SPAM"
  
                    pred_file.write(f"{file_name} {prediction}\n")
