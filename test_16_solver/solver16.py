import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import tokenization
import json
import random

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
        ans_pool = []
        for V in task["question"]['choices']:
            text =  V['text']
            while "  " in text:
                text = text.replace("  ", " ")

            for i in range(10):
                text = text.replace(str(i) + ")", "")
            
            text = text.strip()
            text = text.split(' ')
            var = text

            cnt = 0
            confidence = 0
            for pos_comma in range(1, len(var)):
                res = ""
                fl_good_pos = False
                for word_pos in range(len(var)):
                    if word_pos == pos_comma and var[word_pos] in ["a", 'и', 'или', 'то']:
                        res += " [MASK]"
                        fl_good_pos = True
                    res += ' ' + var[word_pos]
                if not fl_good_pos:
                	continue
                sentence = res
                sentence = sentence.replace(' [MASK] ','[MASK]'); sentence = sentence.replace('[MASK] ','[MASK]'); 
                sentence = sentence.replace(' [MASK]','[MASK]')  # удаляем лишние пробелы
                sentence = sentence.split('[MASK]')
                tokens = ['[CLS]']


                for i in range(len(sentence)):
                    if i == 0:
                        tokens = tokens + self.tokenizer.tokenize(sentence[i]) 
                    else:
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
                vals = np.amax(predicts, axis=-1)
                predicts = np.argmax(predicts, axis=-1)
                predicts = predicts[0][:len(tokens)]    

                out = []
                conf = []
                #idx_comma = self.tokenizer.convert_tokens_to_ids([','])   
                for i in range(len(mask_input[0])):
                    if mask_input[0][i] == 1:
                        if vals[0][i] > 0.2:                   
                            out.append(predicts[i])
                            conf.append(vals[0][i])
                        else:
                            out.append(0)
                            conf.append(vals[0][i])

                out = self.tokenizer.convert_ids_to_tokens(out)          # индексы в текстовые токены
                out = ' '.join(out)                                 # объединяем токены в строку с пробелами
                out = tokenization.printable_text(out)              # в удобочитаемый текст
                out = out.replace(' ##','') 
                if (out[0] == ','):
                    confidence = conf[0]
                    cnt += 1
            if cnt == 1:
                ans.append([confidence, V["id"]])
            if cnt > 1:
                ans_pool.append(V["id"])

        random.shuffle(ans)
        while len(ans) > 2:
            if min(ans)[0] < 0.70:
                ans.remove(min(ans))
            else:
                ans.pop()

        while len(ans) < 2:
            if len(ans_pool) > 0:
                ans.append([0, ans_pool[0]])
                ans_pool.pop(0)
            else:
                val = str(random.randint(1, 5))
                if len(ans) == 0 or val != ans[0][1]:
                    ans.append([0, val])

        ans = sorted(ans, key = lambda x: x[1])
        res = []
        for val in ans:
            res.append(str(val[1]))
        return res


TASK_NUM = 15
obj = Solver()

for i in range(10):
    print("###############" + str(i))
    data = json.load(codecs.open('test_0' + str(i) + '.json', 'r', 'utf-8'))
    inp = data["tasks"][TASK_NUM]
    ans = data["tasks"][TASK_NUM]['solution']
    print(inp['text'])
    print(inp["question"]['choices'])
    print(ans)
    print(obj.predict_from_model(inp))