import re
import os
import json
import codecs
import random
from string import punctuation


class Solver(object):

    def __init__(self, seed=42, data_path='../data/'):
        self.is_train_task = False
        self.seed = seed
        self.init_seed()
        self.dict_emph = {}
        f = open(data_path + "dict.txt", "r", encoding="windows-1251")
        for line in f:
            params = line.split("|")
            self.dict_emph[params[0].strip()] = params

    def init_seed(self):
        random.seed(self.seed)

    def predict(self, task):
        return self.predict_from_model(task)

    def compare_text_with_variants(self, variants, task_type='incorrect'):
        result = ''
        if task_type == 'incorrect':
            for variant in variants:
                if variant not in self.stress:
                    result = variant
        else:
            for variant in variants:
                if variant in self.stress:
                    result = variant
        if not variants:
            return ''
        if not result:
            result = random.choice(variants)
        return result.lower().strip(punctuation)

    def process_task(self, task):
        pass

    def fit(self, tasks):
        pass

    def load(self, path="../data/models/solver4.pkl"):
        pass

    def save(self, path="../data/models/solver4.pkl"):
        pass

    def predict_from_model(self, task):
        words = task["text"].split("\n")[1:]
        ans_is_finded = False
        candidates = []
        for word in words:
            if word.strip() != "":
                mem_word = word.strip().split()[0]
                if word.strip() != "Выпишите это слово.":
                    word = self.kill_brackets(word)
                    res = self.find_word_with_emphasis(word.lower().strip().split()[0])

                    upercase_cnt = 0
                    for sym in res:
                        if sym.isupper():
                            upercase_cnt += 1
                    if upercase_cnt >= 2:
                        candidates.append(word.lower())
                    for mem_sym, res_sym in zip(mem_word, res):
                        if mem_sym.isupper() and res_sym.islower() and not ans_is_finded:
                            return word.lower()
                            ans_is_finded = True
        if len(candidates) == 0:
            candidates = words
        return random.choice(candidates)


##################################################################################
    def kill_brackets(self, word):
        if '(' in word:
            word = re.sub(r"\(.*?\)", "", word)
        return word.strip(' ')


    def find_word_with_emphasis(self, word):
        res_word = ""
        if 'ё' in word:
            for sym in word:
                if sym == 'ё':
                    res_word += 'Ё'
                else:
                    if sym != "'":
                        res_word += sym
            return res_word

        final_word = ""
        res_word = ""
        params = self.dict_emph[word]
        if params[0].strip() == word:
            parts = params[2].strip().split("`")
            word_with_emphasis = ""
            for part in parts:
                word_with_emphasis += part
            res = word_with_emphasis.strip().split("'")
            for part in res[:-1]:
                res_word += part[:-1]
                res_word += part[-1].upper()
            res_word += res[-1]

            if len(final_word) != 0:
                new_final_word = ""
                for f_sym, cur_sym in zip(final_word, res_word):
                    if f_sym.isupper():
                        new_final_word += f_sym
                    else:
                        new_final_word += cur_sym
                final_word = new_final_word
            else:
                final_word = res_word

        if len(final_word) == 0:
            final_word = word
        return final_word

TASK_NUM = 3
obj = Solver()

for i in range(10):
    print("###############" + str(i))
    data = json.load(codecs.open('test_0' + str(i) + '.json', 'r', 'utf-8'))
    inp = data["tasks"][TASK_NUM]
    ans = data["tasks"][TASK_NUM]['solution']
    print(inp['text'])
    print(ans)
    print(obj.predict_from_model(inp))
