import os,re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from NLPyPort.FullPipeline import *
import sys
import stopwords_PT
LANG = "port"

#
#	1-	Tokenization
#	2-	Less than 5 occurrences = <UNK>
#	3-	Numerals <NUM> (originally zero)
#	4-	urls <URL>
#	5-	emojis <EMJ>
#


#########################
#Example for a file input
#########################
# def PTprocessing(text):
NLPort_options = {
			"tokenizer" : True,
			"pos_tagger" : True,
			"lemmatizer" : True,
			"entity_recognition" : True,
			"np_chunking" : True,
			"pre_load" : False,
			"string_or_array" : False
}


bad_POS = ['punc']
if LANG == 'port':
	SW = stopwords_PT.stopwords
elif LANG == 'eng':
	SW = stopwords.words('english')

def check_ok(word,pos):
	if word in SW:
		return False
	if pos in bad_POS:
		return False
	
	return True
		

def remove_stopwords(text,stopwords):
    # print(text)
    words = re.split(";|,|\ |\*|\n|\.|\(|\)|!|W|:",text)
    words_ = list(filter(lambda a: a not in stopwords, words))
    clean = ' '.join(words_)
    return clean

print(os.getcwd())
def tokenization(df):
	tokenized = df['text'].apply(lambda x: re.split(";|,|\ |\*|\n|\.|\(|\)|!|W|:",x))
	return tokenized


def lemmatization_pt(df):

	np.savetxt("temp_text", df['text'].values, fmt='%s')
	# input_file="input_sample.txt"


	text=new_full_pipe("temp_text",options=NLPort_options)
	if(text!=0):
		lema = text.lema
		tokens = text.tokens
		pos = text.pos

		sentence = []
		sentences = []
		for i in range(len(tokens)):
			if "EOS" not in token:
				if check_ok(lema[i],pos[i]):
					sentence.append(token)

			else:
				sentences.append(sentence.copy())
				sentence = []

# input_file="input_sample.txt"


# text=new_full_pipe(input_file,options=NLPort_options)
# if(text!=0):
# 		text.print_conll()

NAME = "labeled_data.csv"
df = pd.read_csv(os.getcwd()+"/"+NAME)
df1 = df[['text', 'class']]
print(df1['text'])

if LANG == "port":
	lemmatization_pt(df1)
quit()
# df1['text'].to_csv("temp_text")




df1['text'].apply(remove_stopwords, args = ([SW])).values
# df1['text'] = df1['text'].apply(remove_stopwords, args = ([SW])).values
df1.loc[:,'text'] = df1['text'].apply(remove_stopwords, args = ([SW])).values

# print(type(clean))
# print(len(clean), df1.shape)
# df1.loc['text'] = clean
# df1.loc["text"] = clean

df1.to_csv("teste-clean_"+NAME)



