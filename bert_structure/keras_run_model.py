# from tensorflow import keras

# from keras.models import Sequential, Model, load_model

# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping,ModelCheckpoint
# from keras.optimizers import Adam
from colorsys import yiq_to_rgb
import os,random
from helper import myDataset, readEmbedding,createMatrix
# from helper import DataHandler, get_accuracy
import keras_models as keras_models
from tensorflow.keras.utils import to_categorical

# from sklearn.model_selection import StratifiedKFold
# from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

import json, datetime, sys
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from datasets import load_metric
# from torch.nn.functional import one_hot
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


class WrongInput(Exception):
	"""Base class for other exceptions"""
	pass

class KerasRunableModel():

	def __init__(self):
		# self.arch = arch

		# self.MAXLEN = 1000
		self.MAX_VOCAB_SIZE=20000
		self.path = os.getcwd()+"/"

		self.OPTMIZING = False
		self.dataset_parts = []
		self.metric = load_metric("accuracy")


	def tokenize_function(self,examples):
		return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=self.config["maxlen"])

	def load_data(self,datafile,mode,split):
		print("Reading data file ...")

		if mode == "kfold":
			for i in range( split):
				print("datafile",datafile)
				data_partial  = myDataset()
				data_partial.readDataset(path="Datasets/{}_{}.csv".format(datafile,i+1))

				self.dataset_parts.append(data_partial)
				print(len(data_partial))

		elif mode == "train_test":
			self.dataset_train  = myDataset()
			self.dataset_train.readDataset(path = "Datasets/{}.csv".format(datafile[0]))
			self.dataset_test  = myDataset()
			self.dataset_test.readDataset(path = "Datasets/{}.csv".format(datafile[1]))


		else:
			base_data = myDataset()
			base_data.readDataset(path = "Datasets/{}.csv".format(datafile))

			train_data, test_data = base_data.split(split = 0.7) 
			# train_data, valid_data = train_data.split(split_ratio = 0.1,random_state = random.seed(42)) 

			self.dataset_train = train_data
			self.dataset_test = test_data

		
	def load_model(self):
		if self.config["model"] == "LSTM":
			self.model = keras_models.LSTM_model(config=self.config,
									embedding_matrix=self.embedding_matrix)
		# self.model.create_model(len(self.TEXT.vocab))
		# self.tokenizer = self.model.tokenizer

	def compute_metrics(self,y_pred,y_real):
		# logits, labels = eval_pred
		# print("******In compute metrics:    ")
		predictions = np.argmax(y_pred.detach().numpy(), axis=-1)
		real = np.argmax(y_real.detach().numpy(), axis=-1)
		# print("Predictions",predictions.shape,real.shape)
		avg = np.average(predictions==real)
		# return avg
		metric_result = self.metric.compute(predictions=predictions, references=real)
		# print("Mine{} tehirs{}".format(avg,metric_result))

		return metric_result

	# def predict(model, sentence):
	# 	tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
	# 	indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
	# 	length = [len(indexed)]                                    #compute no. of words
	# 	tensor = torch.LongTensor(indexed)                          #convert to tensor
	# 	tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
	# 	length_tensor = torch.LongTensor(length)                   #convert to tensor
	# 	prediction = model(length, length_tensor)                  #prediction 
	# 	return prediction.item()  

	def train_test_loop(self,train_data,test_data):
		# max_batch_size = 4
		#
		#	Create val dataset with 10% train
		# train_data, val_data = train_data.split(ratio = 0.1)
		print("train {}\ttest {}".format(len(train_data),len(test_data)))
		print(train_data)
		train_data.preprocessing()
		test_data.preprocessing(tokenizer = train_data.word_tokenizer)
		model = self.model


		X_train = np.array(train_data.text_tokenized)
		y_train = np.array(train_data.labels).astype(int)

		X_test = np.array(test_data.text_tokenized)
		y_test = np.array(test_data.labels).astype(int)


		

		y_train = to_categorical(y_train)
		# train_iterator,val_iterator,test_iterator = torchtext.data.Iterator.splits(
		# 															(train_data,val_data,test_data), 
		# 															batch_size = self.config["batch"],
		# 															sort_within_batch = False,
		# 															sort_key = lambda x: len(x.text))


		#	Define earlystopping
		#	put train test stuff 
		#	Return conf_matrix in array
		training_report = pd.DataFrame(0,index=np.arange(1, self.config['epochs']+1),
										columns=["train_loss","train_acc","val_loss","val_acc"]) #tn, fp, fn, tp)


		rms = optimizers.RMSprop(learning_rate=self.config['lr'])
		model.compile(optimizer=rms, loss = "binary_crossentropy", metrics=['acc'])

		# create checkpoint to save best model, define validation loss as parameter
		filename = self.config['name']+'.model.hdf5'
		checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		# print(train_data.labels)
		# quit()
		# train modeltext_tokenized
		history = model.fit(X_train, 
							y_train,
							epochs=self.config['epochs'], 
							batch_size=self.config['batch'],
							validation_split = 0.1, 
							verbose=1,
							callbacks = checkpoint)

		# save training history in a figure
		training_report["train_loss"] = history.history['loss']
		training_report["train_acc"] = history.history['acc']
		training_report["val_loss"] = history.history['val_loss']
		training_report["val_acc"] = history.history['val_acc']

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.legend(['train','validation'])
		plt.savefig(self.config['name']+'_train.png')
		plt.clf()

		# load best model
		model.load_weights(filename)

		# for epoch in range(self.config["epochs"]):

		# 	train_loss,train_acc = self.model.train(train_iterator,self.compute_metrics)
		# 	# print(train_loss)
		# 	# _,train_acc = self.model.evaluate(train_iterator,self.compute_metrics)
		# 	val_loss,val_acc = self.model.evaluate(val_iterator,self.compute_metrics)
		# 	# print(val_acc)
		# 	training_report.loc[len(training_report)] = [train_loss,train_acc,val_loss,val_acc]
		# 	print("Epoch {}: \tval loss: {},\t val_acc: {}".format(epoch,val_loss,val_acc))
		# print('\n\n')
		# print(test_data[0].__dict__)
		# processed_test_data = [text for text,label in test_iterator]
		# print(processed_test_data[0].size())
		# processed_test_data = torch.tensor(processed_test_data)
		# print(processed_test_data)
		prediction = self.model.predict(X_test)
		# print(test_data['class'])
		# targets = [c for t,c in test_iterator]
		# print(targets)
		# print("targets", targets.shape,prediction.shape)
		# results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
		conf_matrix= confusion_matrix(y_test,prediction[:,0].astype(int)).ravel()




		# y_pred = outputs.predictions.argmax(1)

		return conf_matrix,training_report

	
	

	def return_results(self,results):
		if self.OPTMIZING:
			avg = results[0]
			return avg
		else:
			return results
	def concat_dataset(self,ds_list):
		# ds_concat = ds_list.pop(0)
		X_total = []
		y_total = []
		for ds in ds_list:
			X_total+= ds.text
			y_total+= ds.labels
		# ds_concat = sum(ds_list)
		# list_of_ex = [x for x in ds_concat]
		new_ds = myDataset(X_total,y_total)

		return new_ds

	def run_kfold(self):
		#pra cada k:
		#	pega e junda os datasets
		#	pega o model
		#	chama o train_test_evaluate
		#	add resultados no array/tabelamodel
		print("Running k-fold")
		results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
		log_report = {}
		for k in range(0,self.config['split']):
			print("k = {}/{}".format(k+1,self.config['split']))
			train_parts = self.dataset_parts.copy()
			test_data = train_parts.pop(k)
			train_data = self.concat_dataset(train_parts)
			train_data.preprocessing()
			
			embedding_idx = readEmbedding(self.config['emb_dim'])
			self.embedding_matrix = createMatrix(self.config['emb_dim'],embedding_idx,train_data.word2idx)
			self.load_model()

			conf_matrix,train_log = self.train_test_loop(train_data,test_data)
			
			log_report[k] = train_log
			results_total.loc[len(results_total)] = conf_matrix
		self.model.save()
		return results_total, log_report


	def run_simple(self):

		results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
		log_report = {}
		for k in range(self.config['split']):
			conf_matrix,train_log = self.train_test_loop(self.dataset_train,self.dataset_test)
			log_report[k] = train_log
			results_total.loc[len(results_total)] = conf_matrix
		self.model.save()

		return results_total, log_report

	def run_once(self):
		results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
		log_report = {}
		conf_matrix,train_log = self.train_test_loop(self.dataset_train,self.dataset_test)
		log_report[0] = train_log
		results_total.loc[len(results_total)] = conf_matrix
		self.model.save()

		return results_total, log_report

	def run_turn(self,requirements):
		self.config = requirements

		self.load_data(self.config["dataset"],self.config["mode"],self.config["split"])
		# self.model = self.load_model()
		if self.config["mode"] == "kfold":
			results,train_log = self.run_kfold()
			
		elif self.config["mode"] == "normal":
			results, train_log = self.run_once()

		else: 
			results, train_log = self.run_simple()
		# self.export_results(results)
		return results,train_log




