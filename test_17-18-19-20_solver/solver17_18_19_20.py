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
    	text = task["text"]
    	sentences = text.split('.')[1:]
    	sentence = ""
    	for sen in sentences:
    		sentence += sen + '.'

    	new_sentence = ""
    	cnt = 0
    	for sen in sentence.split("\n"):
    		if (sen != ""):
    			cnt += 1
    			if cnt > 1:
    				if not last in '.!?…':
    					new_sentence += sen[0].lower() + sen[1:]
    				else:
    					new_sentence += sen
    			else:
    				new_sentence += sen
    			last = sen[-1]
    			new_sentence += " "
    		else:
    			if cnt > 0:
    				last = '.'
    	sentence = new_sentence

    	for i in range(10):
    		sentence = sentence.replace("(" + str(i) + ")", "[MASK]")

    	sentence = sentence.split('[MASK]')
    	tokens = ['[CLS]']                           

    	for i in range(len(sentence)):
    	    if i == 0:
    	        tokens = tokens + self.tokenizer.tokenize(sentence[i].strip()) 
    	    else:
    	        tokens = tokens + ['[MASK]'] + self.tokenizer.tokenize(sentence[i].strip())
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
    	out_vals = []

    	for i in range(len(mask_input[0])):
    	    if mask_input[0][i] == 1:                       # [0][i], т.к. сеть возвращает batch с формой (1,512), где в первом элементе наш результат
    	        if vals[0][i] > 0.7:
    	        	out.append(predicts[i]) 
    	        	out_vals.append(vals[0][i])
    	        else:
    	        	out.append(0)
    	        	out_vals.append(vals[0][i])

    	out = self.tokenizer.convert_ids_to_tokens(out)          # индексы в текстовые токены

    	res = []
    	for i in range(len(out)):
    		if out[i] == ',':
    			res.append(str(i + 1))
    	return res


TASK_NUM = 16
obj = Solver()

for i in range(10):
    print("###############" + str(i))
    data = json.load(codecs.open('test_0' + str(i) + '.json', 'r', 'utf-8'))
    inp = data["tasks"][TASK_NUM]
    ans = data["tasks"][TASK_NUM]['solution']
	print(inp['text'])
	print(ans)
    print(obj.predict_from_model(inp))