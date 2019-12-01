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
import tokenization
import json
import random



class Solver(object):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        folder = '../rubert_cased_L-12_H-768_A-12_v2'
        config_path = folder+'/bert_config.json'
        checkpoint_path = folder+'/bert_model.ckpt'
        vocab_path = folder+'/vocab.txt'

        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
        self.model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
        self.model.summary()

        self.option = ["предлог", ["сочинит", "союз"], ["подчинит", "союз"], "местоимен", "наречие", "частица", ["вводное", "слов"], "частица", "союз"]
        self.option_to_list = {
        "предлог": ["в", "без", "до", "из", "к", "на", "по", "о", "от", "перед", "при", "через", "с", "у", "за", "над", "об", "под", "про", "для", "вглубь", "вдоль", "возле", "около", "вокруг", "впереди", "после", "посредством", "путём", "насчёт", "ввиду",  "благодаря", "несмотря"],
        "сочинитсоюз" : ["и", "да", "тоже", "также", "а", "но", "однако", "зато", "же", "или", "либо"], 
        "подчинитсоюз" : ['что', 'чтобы', 'будто', 'когда', 'пока', 'если', 'чтобы', 'дабы', 'хотя', 'хоть', "как", "словно"], 
        "союз" : ["и", "да", "тоже", "также", "а", "но", "однако", "зато", "же", "или", "либо", 'что', 'чтобы', 'будто', 'когда', 'пока', 'если', 'чтобы', 'дабы', 'хотя', 'хоть', "как", "словно"],
        "местоимен" : ["такие", "эти", "себя", "тот", "те", "этот", "таков", "столько", "сам", "самый", "весь", "всякий", "каждый", "иной", "любой", "другой", "некто", 
        "нечто", "некоторый", "кто", "что", "сколько", "чей", "какой", "чем", "кому", "кого", "я", "ты", "он", "она", "оно", "мы", "вы", "они", "себя", "мой", "твой", "свой",
         "ваш", "наш", "его", "её", "их", "кто", "какой", "чей", "где", "который", "откуда", "сколько", "зачем", "кто", "какой", "который", "чей", "сколько", "зачем", "когда", "тот", "этот", "столько", "такой", "таков", "сей", "всякий", "каждый", "самый", "любой", 
         "иной", "другой", "весь", "никто", "ничто", "никакой", "некого", "нечего"], 
         "наречие" : ["вдруг", "вместе", "вовремя", "вовсе", "вперед", "вскоре", "заодно", "затем", "зачастую", "зачем", "кстати", "наверное", "навсегда", "наконец", "отсюда", "оттого", "оттуда", "отчасти", "помимо", "поскольку", "потом", "потому", "поэтому"], 
         "вводноеслов":  ["может", "кажется", "бесспорно", "безусловно", "следовательно"], 
         "частица": ["не", "ни", "нет", "разве", "ли", "да", "вот", "вон", "именно", "ровно", "точно", "лишь", "только", "почти", "исключительно", "как", "даже", "же", "ведь"], 
         " ": 9}



    def init_seed(self):
        random.seed(self.seed)
        
    def load(self, path=""):
        pass

    def save(self, path=""):
        pass

    def fit(self, path=""):
        pass

    def predict_from_model(self, task):
        ans = []
        text = task["text"]
        text = text.replace("?", ".")
        text = text.replace("!", ".")
        text = text.replace("<…>", "[MASK]")
        text = text.replace("<...>", "[MASK]")
        text = text.replace("...",  "[MASK]")
        text = text.replace("…",  "[MASK]")

        key = " "
        for var in self.option:
            if isinstance(var, list):
                if var[0] in text and var[1] in text:
                    key = var[0] + var[1]
                    break
            else:
                if var in text:
                    key = var
                    break

        sentences = text.split('.')
        second_sen = ""
        cnt = 0
        for sen in sentences:
            if "[MASK]" in sen:
                second_sen = sen[4:]
                break
            cnt += 1

        first_sen = sentences[cnt - 1][4:]
        first_sen += '.'
        second_sen += '.'
        sentence = first_sen + ' ' + second_sen


        sentence = sentence.replace(' [MASK] ','[MASK]'); sentence = sentence.replace('[MASK] ','[MASK]'); 
        sentence = sentence.replace(' [MASK]','[MASK]')  # удаляем лишние пробелы
        sentence = sentence.split('[MASK]')
        tokens = ['[CLS]']                           
        upper_case = False
        for i in range(len(sentence)):
            if i == 0:
                tokens = tokens + self.tokenizer.tokenize(sentence[i]) 
            else:
                if tokens[-1] == '.':
                    upper_case = True
                tokens = tokens + ['[MASK]'] + self.tokenizer.tokenize(sentence[i])
        tokens = tokens + ['[SEP]'] 

        token_input = self.tokenizer.convert_tokens_to_ids(tokens)   
        token_input = token_input + [0] * (512 - len(token_input))

        mask_input = [0]*512
        for i in range(len(mask_input)):
            if token_input[i] == 103:
                mask_input[i] = 1

        seg_input = [0]*512
        token_input = np.asarray([token_input])
        mask_input = np.asarray([mask_input])
        seg_input = np.asarray([seg_input])

        predicts = self.model.predict([token_input, seg_input, mask_input])
        predicts = predicts[0]

        if key == " ":

            vals = np.amax(predicts, axis=-1)
            predicts = np.argmax(predicts, axis=-1)
            predicts = predicts[0][:len(tokens)]    
            out = []
            
            for i in range(len(mask_input[0])):
                if mask_input[0][i] == 1:                       # [0][i], т.к. сеть возвращает batch с формой (1,512), где в первом элементе наш результат
                    out.append(predicts[i])

            out = self.tokenizer.convert_ids_to_tokens(out)          # индексы в текстовые токены
            out = ' '.join(out)                                 # объединяем токены в строку с пробелами
            out = tokenization.printable_text(out)              # в удобочитаемый текст
            out = out.replace(' ##','') 
            return out.lower()
        else:
            word_list = self.option_to_list[key]
            new_word_list = []
            if upper_case:
                for word in word_list:
                    new_word_list.append(word[0].upper() + word[1:])
            else:
                new_word_list = word_list
            #print(new_word_list)
            id_word_list = self.tokenizer.convert_tokens_to_ids(new_word_list)
            ID_prob = []
            for i in range(len(mask_input[0])):
                if mask_input[0][i] == 1:
                    for ID in id_word_list:
                        ID_prob.append([predicts[0][i][ID], self.tokenizer.convert_ids_to_tokens([ID])])
            ID_prob = sorted(ID_prob, key = lambda x: x[0], reverse=True)
            return ID_prob[0][1][0].lower()


TASK_NUM = 1
obj = Solver()

for i in range(10):
    print("###############" + str(i))
    data = json.load(codecs.open('test_0' + str(i) + '.json', 'r', 'utf-8'))
    inp = data["tasks"][TASK_NUM]
    ans = data["tasks"][TASK_NUM]['solution']
    print(inp['text'])
    print(ans)
    print(obj.predict_from_model(inp))
