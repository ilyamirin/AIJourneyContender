# coding: utf-8
import sys
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import tokenization

# https://github.com/google-research/bert
# https://github.com/CyberZHG/keras-bert


# папка, куда распаковали преодобученную нейросеть BERT
folder = 'multi_cased_L-12_H-768_A-12'

config_path = folder+'/bert_config.json'
checkpoint_path = folder+'/bert_model.ckpt'
vocab_path = folder+'/vocab.txt'

# создаем объект для перевода строки с пробелами в токены
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)

# загружаем модель
print('Loading model...')
model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
model.summary()          # информация о слоях нейросети - количество параметров и т.д.
print('OK')




# РЕЖИМ 1: предсказание слов, закрытых токеном MASK в фразе. На вход нейросети надо подать фразу в формате: [CLS] Я пришел в [MASK] и купил [MASK]. [SEP]

# входная фраза с закрытыми словами с помощью [MASK]
sentence = 'Я пришел в [MASK] и купил [MASK].'

print(sentence)


#-------------------------
# преобразование в токены (tokenizer.tokenize() не обрабатывает [CLS], [MASK], поэтому добавим их вручную)
sentence = sentence.replace(' [MASK] ','[MASK]'); sentence = sentence.replace('[MASK] ','[MASK]'); sentence = sentence.replace(' [MASK]','[MASK]')  # удаляем лишние пробелы
sentence = sentence.split('[MASK]')             # разбиваем строку по маске
tokens = ['[CLS]']                              # фраза всегда должна начинаться на [CLS]
# обычные строки преобразуем в токены с помощью tokenizer.tokenize(), вставляя между ними [MASK]
for i in range(len(sentence)):
    if i == 0:
        tokens = tokens + tokenizer.tokenize(sentence[i]) 
    else:
        tokens = tokens + ['[MASK]'] + tokenizer.tokenize(sentence[i]) 
tokens = tokens + ['[SEP]']                     # фраза всегда должна заканчиваться на [SEP] 
# в tokens теперь токены, которые гарантированно по словарю преобразуются в индексы
#-------------------------
#print(tokens)

# преобразуем в массив индексов, который можно подавать на вход сети, причем число 103 в нем это [MASK]
token_input = tokenizer.convert_tokens_to_ids(tokens)        
#print(token_input)
# удлиняем до 512 длины
token_input = token_input + [0] * (512 - len(token_input))


# создаем маску, заменив все числа 103 на 1, а остальное 0
mask_input = [0]*512
for i in range(len(mask_input)):
    if token_input[i] == 103:
        mask_input[i] = 1
#print(mask_input)

# маска фраз (вторая фраза маскируется числом 1, а все остальное числом 0)
seg_input = [0]*512


# конвертируем в numpy в форму (1,) -> (1,512)
token_input = np.asarray([token_input])
mask_input = np.asarray([mask_input])
seg_input = np.asarray([seg_input])


# пропускаем через нейросеть...
predicts = model.predict([token_input, seg_input, mask_input])[0]       # в [0] полная фраза с заполненными предсказанными словами на месте [MASK]
predicts = np.argmax(predicts, axis=-1)


# форматируем результат в строку, разделенную пробелами
predicts = predicts[0][:len(tokens)]    # длиной как исходная фраза (чтобы отсечь случайные выбросы среди нулей дальше)
out = []
# добавляем в out только слова в позиции [MASK], которые маскированы цифрой 1 в mask_input
for i in range(len(mask_input[0])):
    if mask_input[0][i] == 1:           # [0][i], т.к. требование было (1,512)
        out.append(predicts[i]) 

out = tokenizer.convert_ids_to_tokens(out)      # индексы в токены
out = ' '.join(out)                             # объединяем в одну строку с пробелами
out = tokenization.printable_text(out)          # в читабельную версию
out = out.replace(' ##','')                     # объединяем раздъединенные слова "при ##шел" -> "пришел"
print('Result:', out)                           # Result: дом его




# РЕЖИМ 2: проверка логичности двух фраз. На вход нейросети надо подать фразу в формате: [CLS] Я пришел в магазин. [SEP] И купил молоко. [SEP]

sentence_1 = 'Я пришел в магазин.'
sentence_2 = 'И купил молоко.'          # Sentence is okey: 99%
#sentence_2 = 'Карась небо Плутон'      # Sentence is okey: 4%


print(sentence_1, '->', sentence_2)

# строки в массивы токенов
tokens_sen_1 = tokenizer.tokenize(sentence_1)
tokens_sen_2 = tokenizer.tokenize(sentence_2)

tokens = ['[CLS]'] + tokens_sen_1 + ['[SEP]'] + tokens_sen_2 + ['[SEP]']
#print(tokens)

# преобразуем строковые токены в числовые индексы:
token_input = tokenizer.convert_tokens_to_ids(tokens)  
# удлиняем до 512      
token_input = token_input + [0] * (512 - len(token_input))

# маска в этом режиме все 0
mask_input = [0] * 512

# в маске предложений под второй фразой, включая конечный SEP, надо поставить 1, а все остальное заполнить 0
seg_input = [0]*512
len_1 = len(tokens_sen_1) + 2                   # длина первой фразы, +2 - включая начальный CLS и разделитель SEP
for i in range(len(tokens_sen_2)+1):            # +1, т.к. включая последний SEP
        seg_input[len_1 + i] = 1                # маскируем вторую фразу, включая последний SEP, единицами
#print(seg_input)


# конвертируем в numpy в форму (1,) -> (1,512)
token_input = np.asarray([token_input])
mask_input = np.asarray([mask_input])
seg_input = np.asarray([seg_input])


# пропускаем через нейросеть...
predicts = model.predict([token_input, seg_input, mask_input])[1]       # в [1] ответ на вопрос, является ли второе предложение логичным по смыслу
#print('Sentence is okey: ', not bool(np.argmax(predicts, axis=-1)[0]), predicts)
print('Sentence is okey:', int(round(predicts[0][0]*100)), '%')                    # [[0.9657724  0.03422766]] - левое число вероятность что второе предложение подходит по смыслу, а правое - что второе предложение случайное


