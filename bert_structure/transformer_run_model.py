# from tensorflow import keras

# from keras.models import Sequential, Model, load_model

# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping,ModelCheckpoint
# from keras.optimizers import Adam
import os

import datasets
# from helper import DataHandler, get_accuracy
import bert_structure.torch_models as torch_models
# from sklearn.model_selection import StratifiedKFold
# from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Value,concatenate_datasets

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

class TransformerRunableModel():

	def __init__(self):
		# self.arch = arch

		self.MAXLEN = 5
		self.MAX_VOCAB_SIZE=20000
		self.path = os.getcwd()+"/"

		self.OPTMIZING = False
		self.dataset_parts = []
		self.metric = load_metric("accuracy")

		

	def tokenize_function(self,examples):
		return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200)

	def one_hot(self,example):	
		example['class'] = [0.0,1.0] if example['class'] else [1.0,0.0] 
		return example
	#	Loads dataset and creates train/test sets
	def load_data(self,datafile,mode,split):
		print("Reading data file ...")

		if mode == "kfold":
			for i in range( split):
				print("datafile",datafile)
				data_partial = load_dataset('csv',data_files = "Datasets/{}_{}.csv".format(datafile,i+1))['train']
				# base = [0]*data_partial['class']

				# print("type",type(data_partial['class'][0]))
				# data_partial= data_partial.map(self.one_hot)
					# Ajusting labels
				new_features = data_partial.features.copy()
				new_features['class'] = datasets.ClassLabel(2,names=["Real","Fake"]) # 
				# # new_features['class'] = Value('float')
				data_partial = data_partial.cast(new_features)
				data_partial = data_partial.rename_column("class", "labels")

				data_tokenized = data_partial.map(self.tokenize_function,batched=False)
				data_tokenized.set_format(type=data_tokenized.format["type"], columns=['labels','input_ids','token_type_ids','attention_mask'])

				self.dataset_parts.append(data_tokenized)

		elif mode == "train_test":
			data_train = load_dataset('csv',data_files = "Datasets/{}.csv".format(datafile[0]))['train']
			# print("DATA",data_train[0])
			new_features = data_train.features.copy()
			print(data_train.features)

			new_features['class'] = datasets.ClassLabel(2,names=["Real","Fake"]) # 
			data_train = data_train.cast(new_features)
			data_train = data_train.rename_column("class", "labels")
			data_tokenized = data_train.map(self.tokenize_function,batched=False)
			self.dataset_train = data_tokenized

			data_test = load_dataset('csv',data_files = "Datasets/{}.csv".format(datafile[1]))['train']
			new_features = data_test.features.copy()
			new_features['class'] = datasets.ClassLabel(2,names=["Real","Fake"]) # 
			data_test = data_test.cast(new_features)
			data_test = data_test.rename_column("class", "labels")
			data_tokenized = data_test.map(self.tokenize_function,batched=False)
			self.dataset_test = data_tokenized

		else:

			data_total = load_dataset('csv',data_files = "Datasets/{}.csv".format(datafile))['train']
			new_features = data_total.features.copy()
			new_features['class'] = Value('float')
			data_total = data_total.cast(new_features)
	

			data_total = data_total.rename_column("class", "labels")

			data_tokenized = data_total.map(self.tokenize_function,batched=False)
			# self.data_complete = data_tokenized
			data_split = data_tokenized.train_test_split(self.config['split'])
			self.dataset_train = data_split['train']
			self.dataset_test = data_split['test']

			
		
	def load_model(self):
		if self.config["model"] == "bertimbau":
			self.model = torch_models.BERTimbau(self.config)
		elif self.config["model"] == "multilingual":
			self.model = torch_models.MultilingualBERT(self.config)
		self.model.create_model()
		self.tokenizer = self.model.tokenizer

	def compute_metrics(self,eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)
		print("Predictions",eval_pred)
		avg = np.average(predictions==labels)
		# return avg
		metric_result = self.metric.compute(predictions=predictions, references=labels)
		print("Mine{} tehirs{}".format(avg,metric_result))

		return metric_result

	def train_test_loop(self,model_obj,lr,train_data,test_data):
		max_batch_size = 4
		#
		#	Create val dataset with 10% train
		data_split = train_data.train_test_split(0.1)
		train_data = data_split['train']
		val_data = data_split['test']
		print("train {}\tval {}".format(len(train_data),len(val_data)))
		print(train_data)
		#	Define earlystopping
		#	put train test stuff 
		#	Return conf_matrix in array
		
		training_args = TrainingArguments(
			# label_names=["label"],
			output_dir = "./results", 
			do_train = True,
			do_eval = True,
			evaluation_strategy = "steps",
			save_strategy = "steps",
			logging_dir='./logs',
			logging_strategy= "steps",
			logging_steps=400,
			eval_steps=400,
			save_steps= 400,
			num_train_epochs = self.config["epochs"],
			learning_rate=self.config["lr"],
			weight_decay=0.01,
			load_best_model_at_end=True,
			metric_for_best_model = "accuracy",
			greater_is_better= True,
			#Performance part
			per_device_train_batch_size=max_batch_size,
			per_device_eval_batch_size=max_batch_size,
			)
		print("TRAINDATA",train_data.features)

		print("VALDATA",val_data.features)
		trainer = torch_models.MyTrainer(
			model=self.model.model, 
			args=training_args, 
			train_dataset=train_data, 
			eval_dataset=val_data,
			compute_metrics = self.compute_metrics
		)
		print(trainer.train_dataset[0])

		trainer.train()


		# try:
		# 	trainer.train()
		# except Exception as e:
		# 	print(trainer.train_dataset[0])

		# 	for batch in trainer.get_train_dataloader():
		# 		print({k: v.shape for k, v in batch.items()})

		# 	print(e)
		# 	quit()


		outputs = trainer.predict(test_data)
		trainer.save_model("./models/model")

		y_pred = outputs.predictions.argmax(1)
		print("y_pred",outputs.predictions)
		y_true = test_data['labels'] 
		conf_matrix= confusion_matrix(y_true,y_pred).ravel()
		# print(conf_matrix)
		# conf_matrix = conf_matrix.ravel()
		
		train_log = trainer.state.log_history

		return conf_matrix,train_log

	
	

	def return_results(self,results):
		if self.OPTMIZING:
			avg = results[0]
			return avg
		else:
			return results
	
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
			train_data = concatenate_datasets(train_parts)

			conf_matrix,train_log = self.train_test_loop(self.model,self.config['lr'],train_data,test_data)
			# conf_matrix,train_log = ([1,1,1,1],{'s':5})
			log_report[k] = train_log
			results_total.loc[len(results_total)] = conf_matrix
		return results_total, log_report

	def run_simple(self):

		results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
		log_report = {}
		for k in range(self.config['split']):
			conf_matrix,train_log = self.train_test_loop(self.model,self.config['lr'],self.dataset_train,self.dataset_test)
			log_report[k] = train_log
			results_total.loc[len(results_total)] = conf_matrix
		return results_total, log_report

	def run_once(self):
		results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
		log_report = {}
		conf_matrix,train_log = self.train_test_loop(self.model,self.config['lr'],self.dataset_train,self.dataset_test)
		log_report[0] = train_log
		results_total.loc[len(results_total)] = conf_matrix
		return results_total, log_report

	def run_turn(self,requirements):
		self.config = requirements
		self.load_model()
		self.model.model = self.model.model.to(self.config["device"])
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




