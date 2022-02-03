#https://huggingface.co/docs/transformers/training

import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split  

class DataHandler(Dataset):

	def __init__(self):
		self.sentences = None
		self.labels_array = None
		
		# self.readDataFile(filename)

	def populate(self,X,Y):
		self.sentences = X
		self.labels_array = Y

	def __len__(self):
		return len(self.sentences)
	
	def __getitem__(self,idx):
		text = self.sentences[idx]
		target = self.labels_array[idx]
		return {'text':text, 'label':target}

	def readDataFile(self,filename):

		path = os.getcwd()
		df = pd.read_csv(path+'/Datasets/'+filename)
		df["text"] = df["text"].astype(str)
		labels_words = df["class"].values
		classes = np.unique(labels_words).tolist()
		print("Classes found: ",classes)
		full_arraya=np.ndarray((len(labels_words),0))
		for cl in classes:
			cl_array = (labels_words==cl).astype(int)
			full_arraya = np.column_stack((full_arraya,cl_array))
		self.labels_array = full_arraya

		self.int_encoding = np.argmax(self.labels_array,axis = 0)

		# print("arr",self.labels_array.shape)
		self.sentences = df["text"].values.tolist()

	def split(self,rate):
		X_train, X_test, y_train, y_test = train_test_split( self.sentences, self.labels_array, test_size=rate, random_state=42)
		train_split = DataHandler()
		train_split.populate(X_train,y_train)
		test_split = DataHandler()
		test_split.populate(X_test,y_test)

		return train_split,test_split
		
	def map(self,function):
		new_X = list(map(function, self.sentences))
		response = DataHandler()
		response.populate(new_X,self.labels_array)
		return response
	def apply(self,function):
		new_X = function(self.sentences)
		print(new_X)
		response = DataHandler()
		response.populate(new_X,self.labels_array)
		return response
# class DataHandler():

# 	def __init__(self):
# 		self.sentences = None
# 		self.labels_array = None
# 	def readDataFile(self,filename):

# 		path = os.getcwd()
# 		df = pd.read_csv(path+'/Datasets/'+filename)
# 		df["text"] = df["text"].astype(str)
# 		labels_words = df["class"].values
# 		classes = np.unique(labels_words).tolist()
# 		print("Classes found: ",classes)
# 		full_arraya=np.ndarray((len(labels_words),0))
# 		for cl in classes:
# 			cl_array = (labels_words==cl).astype(int)
# 			full_arraya = np.column_stack((full_arraya,cl_array))
# 		self.labels_array = full_arraya
# 		# print(self.labels_array)

# 		self.int_encoding = np.argmax(self.labels_array,axis = 0)

# 		# print("df",df.shape)
# 		print("arr",self.labels_array.shape)



# 		self.sentences = df["text"].values.tolist()

	


# dh = DataHandler()
# dh.readDataFile("fakebr.csv")