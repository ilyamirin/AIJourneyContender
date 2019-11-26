import re
import random
import pymorphy2
import json
import codecs

class Solver(object):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.symbols = [chr(i) for i in range(ord('а'), ord('я') + 1)]
        self.dictionary = {}
        #############################################
        #Словарь 1
        #f = open("data/dict.txt", "r", encoding="windows-1251")
        #for line in f:
        #   res_word = ""
        #   params = line.split("|")
        #   self.dictionary[params[0].strip().replace("'", "")] = True 

        #Словарь 2
        f = open("../data/russian.txt", "r", encoding="windows-1251")
        for line in f:
            self.dictionary[line.strip()] = True
        #############################################

    def init_seed(self):
        return random.seed(self.seed)

    def predict_from_model(self, task):
        TASK_TEXT, variants = task["text"].split("\n")[0], task["text"].split("\n")[1:]
        if "на месте пропуска пишется буква" in TASK_TEXT:
            return self.known_letter(TASK_TEXT, task)        
        else:
            return self.variants_out_of_text(task, self.symbols)


    def load(self, path=""):
        pass

    def save(self, path=""):
        pass

    def fit(self, path=""):
        pass


    def find_word(self, word):
        return (word in self.dictionary)

    def kill_brackets(self, word):
        if '(' in word:
            word = re.sub(r"\(.*?\)", "", word)
        for i in ['(', ')'] + [str(i) for i in range(10)]:
            word = word.replace(i, "")
        return word.strip(' ')

    def known_letter(self, TASK_TEXT, task):
        for pos in range(0, len(TASK_TEXT)):
            if TASK_TEXT[pos].isupper() and TASK_TEXT[pos + 1] == '.':
                sym = TASK_TEXT[pos].lower()
        return self.variants_out_of_text(task, [sym])


    def variants_out_of_text(self, task, symbols):
        ans_pool = []
        for variant in task["question"]["choices"]:
            words = variant["text"].split(', ')
            for sym in symbols:
                cnt = 0
                cnt_good = 0
                for word in words:
                    word = self.kill_brackets(word)
                    word = word.strip()
                    cnt += 1
                    if self.find_word(word.replace("..", sym)):
                        cnt_good += 1
                if cnt == cnt_good:
                    ans_pool.append(variant["id"])
                    break
        return ans_pool

obj = Solver()

for i in range(10):
    data = json.load(codecs.open('test_0' + str(i) + '.json', 'r', 'utf-8'))
    inp = data["tasks"][9]
    ans = data["tasks"][9]['solution']
    print(ans)

    print(obj.predict_from_model(inp))