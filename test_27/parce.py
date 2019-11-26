import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import string
import random
from keras_bert import load_trained_model_from_checkpoint
import tokenization
import numpy as np

f_in = open('essay_themes_pool.txt', "r", encoding="utf-8")
f_out = open("generated_text.txt", "w", encoding="utf-8")

folder = 'rubert_cased_L-12_H-768_A-12_v2'
config_path = folder+'/bert_config.json'
checkpoint_path = folder+'/bert_model.ckpt'
vocab_path = folder+'/vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
model.summary()

def find_author(text):
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


all_text = ""
for line in f_in:
	all_text += line
texts = all_text.split("****")
ind = 0
for text in texts:
	print(ind)
	ind += 1
	for sym in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'.upper():
		text = text.replace(sym + '.', sym)
	text = text.replace('–', ' – ')
	for i in range(10):
		text = text.replace('\n\n', "\n")
		text = text.replace('!', ".")
		text = text.replace('?', ".")
		text = text.replace("  ", " ")
	chap_cnt = 0
	f_out.write("****")
	for chap in text.split('\n'):
		f_out.write("\n")
		sentences = chap.split('.')
		cnt = 0
		last = ""
		last = sentences[-1]
		for sen in sentences:
			cnt += 1
			if cnt == 1 or sen == last:
				f_out.write(sen.replace(" .", '.').replace(" ,", ","))
				continue
			sentence = sen
			sentence = sentence.replace(',', ' ,')
			sentence = sentence.split(' ')

			for i in range(2):
				rand_ind = random.randrange(len(sentence))
				if len(sentence[rand_ind]) > 1:
					sentence.insert(rand_ind, "[MASK]") 
				else:
					i -= 1
					continue
				mem_sentence = sentence
				sentence =  ' '.join(sentence)
				sentence = sentence.replace(' [MASK] ','[MASK]'); sentence = sentence.replace('[MASK] ','[MASK]'); 
				sentence = sentence.replace(' [MASK]','[MASK]')  # удаляем лишние пробелы
				sentence = sentence.split('[MASK]')

				tokens = ['[CLS]']                           
				for i in range(len(sentence)):
				    if i == 0:
				    	tokens = tokens + tokenizer.tokenize(sentence[i]) 
				    else:
				    	tokens = tokens + ['[MASK]'] + tokenizer.tokenize(sentence[i])
				tokens = tokens + ['[SEP]'] 
				token_input = tokenizer.convert_tokens_to_ids(tokens)   
				token_input = token_input + [0] * (512 - len(token_input))

				mask_input = [0]*512
				for i in range(len(mask_input)):
				    if token_input[i] == 103:
				        mask_input[i] = 1

				seg_input = [0]*512
				token_input = np.asarray([token_input])
				mask_input = np.asarray([mask_input])
				seg_input = np.asarray([seg_input])

				predicts = model.predict([token_input, seg_input, mask_input])
				predicts = predicts[0]

				predicts = np.argmax(predicts, axis=-1)
				predicts = predicts[0][:len(tokens)]    
				out = []
				
				for i in range(len(mask_input[0])):
				    if mask_input[0][i] == 1:                       # [0][i], т.к. сеть возвращает batch с формой (1,512), где в первом элементе наш результат
				        out.append(predicts[i])
				        break
				out = tokenizer.convert_ids_to_tokens(out)
				for i in range(len(mem_sentence)):
					if mem_sentence[i] == "[MASK]":
						mem_sentence[i] = out[0]
				sentence = mem_sentence
			f_out.write(' '.join(sentence).replace(" .", '.').replace(" ,", ","))
			f_out.write('. ')