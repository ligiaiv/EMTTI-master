from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


class myDataset:
	def __init__(self,text=[],
					labels = [],
					MAXLEN = 2000,
					MAX_VOCAB_SIZE = 20000):

		self.text = text
		self.labels = labels
		self.MAXLEN = MAXLEN
		self.MAX_VOCAB_SIZE = MAX_VOCAB_SIZE

	def __len__(self):
		return len(self.text)

	def readDataset(self,path):
		# print("READING DATASETS")
		df = pd.read_csv(path)
		# print(df)
		self.labels = df["class"].tolist()

		self.text = df["text"].tolist()

	def preprocessing(self,tokenizer = None):
	  
		if tokenizer:
			self.word_tokenizer = tokenizer
		else:
			self.word_tokenizer = Tokenizer(num_words=self.MAX_VOCAB_SIZE)   
			
		self.word_tokenizer.fit_on_texts(self.text) #gives each word a number
		self.word2idx = self.word_tokenizer.word_index
		self.text_tokenized = sequence.pad_sequences(self.word_tokenizer.texts_to_sequences(self.text),maxlen=self.MAXLEN)

	def split(self,ratio):
		text_train, text_test, y_train, y_test = train_test_split(self.text, self.labels, 
													test_size=ratio, random_state=42)
		train_dataset = myDataset(text_train,y_train)
		test_dataset = myDataset(text_test,y_test)
		
		return train_dataset,test_dataset


def readEmbedding(emb_dim):
	embeddings_index = {}
	EMBEDDING_DIR = "Embeddings/"
	f = open(EMBEDDING_DIR+"glove_s{}.txt".format(emb_dim))
	print("Reading embeddings...")
	for line in f:
		try:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		except:
			continue
	f.close()

	print('Found %s word vectors.' % len(embeddings_index))

	return embeddings_index

def createMatrix(embedding_dim,embeddings_index,word2idx):
	print("Creating embedding matrix...")
	embedding_matrix = np.zeros((len(word2idx) + 1, embedding_dim))
	for word, i in word2idx.items():
		embedding_vector = embeddings_index.get(word)

		if embedding_vector is not None:
			# print(embedding_vector.shape)
			# if len(embedding_vector)>100:
			# 	print(embedding_vector)
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector[-100:]
	return embedding_matrix