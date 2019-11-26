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
import json
import random
from summa.summarizer import summarize
from keras_bert import extract_embeddings
from semantic_text_similarity.models import WebBertSimilarity
from semantic_text_similarity.models import ClinicalBertSimilarity
import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import tokenization
import pymorphy2
morph = pymorphy2.MorphAnalyzer()



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

	def kill_trash(self, text):
		kill_it = ["Напишите сочинение по прочитанному тексту.",
		"Сформулируйте одну из проблем, поставленных автором текста.",
		"Прокомментируйте сформулированную проблему.",
		"Включите в комментарий два примера-иллюстрации из прочитанного текста, которые, по Вашему мнению, важны для понимания проблемы исходного текста (избегайте чрезмерного цитирования).",
		"Поясните значение каждого примера и укажите смысловую связь между ними.",
		"Сформулируйте позицию автора (рассказчика).",
		"Выразите своё отношение к позиции автора по проблеме исходного текста (согласие или несогласие) и обоснуйте его.",
		"Объём сочинения — не менее 150 слов.",
		"Объём сочинения – не менее 150 слов.",
		"Работа, написанная без опоры на прочитанный текст (не по данному тексту), не оценивается.",
		"Если сочинение представляет собой пересказанный или полностью переписанный исходный текст без каких бы то ни было комментариев, то такая работа оценивается 0 баллов.",
		"Сочинение пишите аккуратно, разборчивым почерком.",
		"Прочитайте текст и выполните задание.",
		"Если сочинение представляет собой пересказанный или полностью переписанный исходный текст без каких бы то ни было комментариев, то такая работа оценивается"]
		for sen in kill_it:
			text = text.replace(sen, "")
		for i in range(200):
			text = text.replace('(' + str(i) + ')', ' ')
		text = text.replace("  ", " ")
		return text

	def find_author(self, text):
		TEXT = ""
		for T in text.split('\n'):
			if T != "":
				TEXT += T + "\n"
		first_part = TEXT.split("\n")[0].strip().split(' ')
		for i in range(len(first_part) - 2):
			if first_part[i][0].isupper() and first_part[i + 1][0].isupper() and first_part[i + 2][0].isupper():
				return first_part[i] + ' ' + first_part[i + 1] + ' ' + first_part[i + 2]
		for i in range(len(first_part) - 1):
			if first_part[i][0].isupper() and first_part[i + 1][0].isupper():
				return first_part[i] + ' ' + first_part[i + 1]
		return "NO"

	def predict_from_model(self, task):
		ans = []
		text = task["text"]
		text = self.kill_trash(text)
		text_full = text
		text1 = text
		text2 = summarize(text, language='russian', ratio=0.07)
		#print(text2)

		sentence_1 = text2

		f_in = open('essay_themes_pool.txt', "r", encoding="utf-8")
		f_in_new = open('New_essays_out.txt', "r", encoding="utf-8")

		new_all_text = ""
		for line in f_in_new:
			new_all_text += line

		all_text = ""
		for line in f_in:
			all_text += line
		texts = all_text.split("****")
		texts_new = new_all_text.split("****")

		ind = 0
		best = 0
		for text, new_text in zip(texts, texts_new):
			text = text.strip('\n')
			text = text.strip(' ')
			new_text = new_text.strip('\n')
			new_text = new_text.strip(' ')
			for i in range(1000):
				text = text.replace("\n\n", "\n")
			text = text.split('\n')
			new_text = new_text.split('\n')
			for ind in range(1, len(text) - 1):
				text[ind] = new_text[ind]
	
			text = '\n'.join(text)
			#print(text)
			ind += 1
			sentence_2 = text.split('\n')[0] + ' ' + text.split('\n')[-1]
			#sentence_2 = summarize(text, language='russian', ratio=0.1)

			tokens_sen_1 = self.tokenizer.tokenize(sentence_1)[:250]
			tokens_sen_2 = self.tokenizer.tokenize(sentence_2)[:250]
			tokens = ['[CLS]'] + tokens_sen_1 + ['[SEP]'] + tokens_sen_2 + ['[SEP]']
			token_input = self.tokenizer.convert_tokens_to_ids(tokens)      
			token_input = token_input + [0] * (512 - len(token_input))
			mask_input = [0] * 512
			seg_input = [0]*512
			len_1 = len(tokens_sen_1) + 2                   # длина первой фразы, +2 - включая начальный CLS и разделитель SEP
			for i in range(len(tokens_sen_2)+1):            # +1, т.к. включая последний SEP
			    seg_input[len_1 + i] = 1                # маскируем вторую фразу, включая последний SEP, единицами

			token_input = np.asarray([token_input])
			mask_input = np.asarray([mask_input])
			seg_input = np.asarray([seg_input])
			predicts = self.model.predict([token_input, seg_input, mask_input])[1]
			if (predicts[0][0] > best):
				#print("ESSAY")
				#print(sentence_2)
				best_essay = text
				best_essay = best_essay.replace("\n\n", "\n")
				best_essay = best_essay.strip('\n')
				best_essay = best_essay.strip(' ')

				best = predicts[0][0]
				#print('Sentence is okey:', int(round(predicts[0][0]*100)), '%')

		KILL_AUTHOR = self.find_author(best_essay)
		#print(KILL_AUTHOR)

		name_inf = ""
		if KILL_AUTHOR != None:
			p = morph.parse(KILL_AUTHOR.strip('.').split(' ')[-1])[0] 
			name_inf = p.normal_form[0].upper() + p.normal_form[1:]
			KILL_AUTHOR = KILL_AUTHOR.replace(KILL_AUTHOR.split(' ')[-1], name_inf)
			best_essay = best_essay.replace(KILL_AUTHOR, "автор")
		sents = best_essay.split('\n')
		
		result = sents[0] + "\n" + 'Автор выражает своё мнение о поставленно проблему в следующих строках своего текста:"' + " ".join(sentence_1.split('\n')) + "'" + '\n'
		for sen in sents[2:]:
			result += sen + '\n'

		return result


TASK_NUM = 26
obj = Solver()

for i in range(10):
    data = json.load(codecs.open('test_0' + str(i) + '.json', 'r', 'utf-8'))
    inp = data["tasks"][TASK_NUM]
    ans = data["tasks"][TASK_NUM]['solution']
    print(ans)
    print(obj.predict_from_model(inp))