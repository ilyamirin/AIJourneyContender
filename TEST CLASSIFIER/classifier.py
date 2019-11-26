from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.toktok import ToktokTokenizer
import random
import numpy as np
from sklearn.svm import LinearSVC
#import utils
import os
from utils import read_config, load_pickle, save_pickle
import random
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from string import punctuation
import joblib
import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import json
import random



class Solver(object):

    def __init__(self, seed=42, ngram_range=(1, 3)):
        self.seed = seed
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        self.clf = LinearSVC(multi_class='ovr')
        self.init_seed()
        self.word_tokenizer = ToktokTokenizer()

    def init_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def predict(self, task):
        return self.predict_from_model(task)

    def fit(self, tasks):
        texts = []
        classes = []
        for data in tasks:
            for task in data:
                idx = int(task["id"])
                text = "{} {}".format(" ".join(self.word_tokenizer.tokenize(task['text'])), task['question']['type'])
                texts.append(text)
                classes.append(idx)
        vectors = self.vectorizer.fit_transform(texts)
        classes = np.array(classes)
        self.classes = np.unique(classes)
        self.clf.fit(vectors, classes)
        return self

    def predict_from_model(self, task):
        texts = []
        for task_ in task:
            text = "{} {}".format(" ".join(self.word_tokenizer.tokenize(task_['text'])), task_['question']['type'])
            texts.append(text)
        return self.clf.predict(self.vectorizer.transform(texts))
    
    def fit_from_dir(self, dir_path):
        tasks = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".json"):
                data = read_config(os.path.join(dir_path, file_name))
                tasks.append(data)
        return self.fit(tasks)
    
    @classmethod
    def load(cls, path):
        return load_pickle(path)
    
    def save(self, path):
        save_pickle(self, path)


obj = Solver()

train_path='public_set/train'

tasks = []
for filename in os.listdir(train_path):
    if filename.endswith(".json"):
        data = read_config(os.path.join(train_path, filename))
        tasks.append(data)
print("Fitting Classifier...")
obj.fit(tasks)
print("Classifier is ready!")   

for i in range(10):
    #for task in range(27):
    data = json.load(codecs.open('test_0' + str(i) + '.json', 'r', 'utf-8'))
    inp = data["tasks"]
    print(obj.predict_from_model(inp))