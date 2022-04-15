# from tensorflow import keras

# from keras.models import Sequential, Model, load_model

# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping,ModelCheckpoint
# from keras.optimizers import Adam
import os,random

# from helper import DataHandler, get_accuracy
import models
# from sklearn.model_selection import StratifiedKFold
# from sklearn.utils.class_weight import compute_class_weight
import torchtext.legacy as torchtext

from sklearn.metrics import confusion_matrix

import json, datetime, sys
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from datasets import load_metric
from torch.nn.functional import one_hot
import torch
class WrongInput(Exception):
	"""Base class for other exceptions"""
	pass

class TorchRunableModel():

	def __init__(self):
		# self.arch = arch

		self.MAXLEN = 1000
		self.MAX_VOCAB_SIZE=20000
		self.path = os.getcwd()+"/"

		self.OPTMIZING = False
		self.dataset_parts = []
		self.metric = load_metric("accuracy")

		self.TEXT = torchtext.data.Field(tokenize = 'spacy', tokenizer_language = 'pt_core_news_sm')
		# self.TEXT = torchtext.data.Field(tokenize = , tokenizer_language = 'pt_core_news_sm')

		self.LABEL = torchtext.data.LabelField(dtype = torch.int)
		self.fields = [('text', self.TEXT), ('class', self.LABEL)]

	def tokenize_function(self,examples):
		return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=self.MAXLEN)

	def load_data(self,datafile,mode,split):
		print("Reading data file ...")

		if mode == "kfold":
			for i in range( split):
				print("datafile",datafile)
				data_partial  = torchtext.data.TabularDataset(
					path="Datasets/{}_{}.csv".format(datafile,i+1),
					format='csv',
					fields=self.fields,
					skip_header=True
					)

				# print(data_partial[0].__dict__)
				self.dataset_parts.append(data_partial)

		elif mode == "train_test":
			# data_train = load_dataset('csv',data_files = "Datasets/{}.csv".format(datafile[0]))['train']
			# print("DATA",data_train[0])
			self.dataset_train  = torchtext.data.TabularDataset(
					path="Datasets/{}.csv".format(datafile[0]),
					format='csv',
					fields=self.fields,
					skip_header=True
					)
			self.dataset_test  = torchtext.data.TabularDataset(
					path="Datasets/{}.csv".format(datafile[0]),
					format='csv',
					fields=self.fields,
					skip_header=True
					)

		else:



			base_data  = torchtext.data.TabularDataset(
					path="Datasets/{}.csv".format(datafile),
					format='csv',
					fields=self.fields,
					skip_header=True
					)
			train_data, test_data = base_data.split(split_ratio=0.7,random_state = random.seed(42)) 
			# train_data, valid_data = train_data.split(split_ratio = 0.1,random_state = random.seed(42)) 

			self.dataset_train = train_data
			self.dataset_test = test_data

		
	def load_model(self):
		if self.config["model"] == "LSTM":
			self.model = models.LSTMgeneral(self.config)
		self.model.create_model(len(self.TEXT.vocab))
		# self.tokenizer = self.model.tokenizer

	def compute_metrics(self,y_pred,y_real):
		# logits, labels = eval_pred
		predictions = np.argmax(y_pred, axis=-1)
		real = np.argmax(y_real, axis=-1)
		print("Predictions",predictions.shape,real.shape)
		avg = np.average(predictions==real)
		# return avg
		metric_result = self.metric.compute(predictions=predictions, references=real)
		print("Mine{} tehirs{}".format(avg,metric_result))

		return metric_result

	def train_test_loop(self,train_data,test_data):
		max_batch_size = 4
		#
		#	Create val dataset with 10% train
		train_data, val_data = train_data.split(split_ratio = 0.1,random_state = random.seed(42))
		print("train {}\tval {}".format(len(train_data),len(val_data)))
		print(train_data)

		train_iterator,val_iterator = torchtext.data.BucketIterator.splits((train_data,val_data), batch_size = self.config["batch"])


		#	Define earlystopping
		#	put train test stuff 
		#	Return conf_matrix in array
		training_report = pd.DataFrame(columns=["train_loss","train_acc","val_loss","val_acc"]) #tn, fp, fn, tp)

		self.TEXT.build_vocab(train_data)
		self.LABEL.build_vocab(train_data)

		self.load_model()
		self.model.model = self.model.model.to(self.config["device"])



		for epoch in range(self.config["epochs"]):

			train_loss = self.model.train(train_iterator,self.compute_metrics)
			print(train_loss)
			_,train_acc = self.model.evaluate(train_iterator,self.compute_metrics)
			val_loss,val_acc = self.model.evaluate(val_iterator,self.compute_metrics)
			print(val_acc)
			training_report.loc[len(training_report)] = [train_loss,train_acc,val_loss,val_acc]
			print("Epoch {}: \tval loss: {},\t val_acc: {}".format(epoch,val_loss,val_acc))
		print('\n\n')
		prediction = self.model.model(test_data)
		results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
		conf_matrix= confusion_matrix(test_data.targets,prediction).ravel()


		torch.save(self.model.model.state_dict(), "./models/model_LSTM")


		# y_pred = outputs.predictions.argmax(1)

		return conf_matrix,training_report

	
	

	def return_results(self,results):
		if self.OPTMIZING:
			avg = results[0]
			return avg
		else:
			return results
	def concat_tabulardataset(self,ds_list):
		ds_concat = ds_list.pop(0)
		for ds in ds_list:
			ds_concat+= ds
		# ds_concat = sum(ds_list)
		list_of_ex = [x for x in ds_concat]
		new_ds = torchtext.data.Dataset(list_of_ex,self.fields)

		return new_ds

	def run_kfold(self):
		#pra cada k:
		#	pega e junda os datasets
		#	pega o model
		#	chama o train_test_evaluate
		#	add resultados no array/tabelamodel
		results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
		log_report = {}
		for k in range(self.config['split']):
			train_parts = self.dataset_parts.copy()
			test_data = train_parts.pop(k)
			train_data = self.concat_tabulardataset(train_parts)

			conf_matrix,train_log = self.train_test_loop(train_data,test_data)
			# conf_matrix,train_log = ([1,1,1,1],{'s':5})
			log_report[k] = train_log
			results_total.loc[len(results_total)] = conf_matrix
		return results_total, log_report

	def run_simple(self):

		results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
		log_report = {}
		for k in range(self.config['split']):
			conf_matrix,train_log = self.train_test_loop(self.dataset_train,self.dataset_test)
			log_report[k] = train_log
			results_total.loc[len(results_total)] = conf_matrix
		return results_total, log_report

	def run_once(self):
		results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
		log_report = {}
		conf_matrix,train_log = self.train_test_loop(self.dataset_train,self.dataset_test)
		log_report[0] = train_log
		results_total.loc[len(results_total)] = conf_matrix
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




