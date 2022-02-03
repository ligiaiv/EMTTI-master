# from tensorflow import keras

# from keras.models import Sequential, Model, load_model

# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping,ModelCheckpoint
# from keras.optimizers import Adam
import os
# from helper import DataHandler, get_accuracy
import models
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


class WrongInput(Exception):
    """Base class for other exceptions"""
    pass

class RunableModel():

	def __init__(self):
		# self.arch = arch

		self.MAXLEN = 40
		self.MAX_VOCAB_SIZE=20000
		self.path = os.getcwd()+"/"

		self.OPTMIZING = False
		self.dataset_parts = []
		self.metric = load_metric("accuracy")

		

	def tokenize_function(self,examples):
			return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200)


	#	Loads dataset and creates train/test sets
	def load_data(self,datafile,kfold,split):
		print("Reading data file ...")

		if kfold:
			for i in range( split):
				print("datafile",datafile)
				data_partial = load_dataset('csv',data_files = "Datasets/{}_{}.csv".format(datafile,i+1))['train']
				#	Ajusting labels
				new_features = data_partial.features.copy()
				new_features['class'] = Value('float')
				data_partial = data_partial.cast(new_features)

				data_partial = data_partial.rename_column("class", "labels")

				data_tokenized = data_partial.map(self.tokenize_function,batched=False)
				
				self.dataset_parts.append(data_tokenized)

			
		else:

			data_total = load_dataset('csv',data_files = "Datasets/{}.csv".format(datafile))['train']
			new_features = data_total.features.copy()
			new_features['class'] = Value('float')
			data_total = data_total.cast(new_features)
	

			data_total = data_total.rename_column("class", "labels")

			data_tokenized = data_total.map(self.tokenize_function,batched=False)
			self.data_complete = data_tokenized

			
		
	def load_model(self):
		if self.config["model"] == "bertimbau":
			self.model = models.BERTimbau(self.config)
			self.tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False,is_split_into_words=True)


	def compute_metrics(self,eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)
		return self.metric.compute(predictions=predictions, references=labels)

	def train_test_loop(self,model_obj,lr,train_data,test_data):
		batch_size = 64
		epochs = 20
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
			output_dir = "./results", 
			do_train = True,
			do_eval = True,
			evaluation_strategy = "epoch",
			save_strategy = "epoch",
			logging_strategy= "steps",
			logging_dir= "logs",
			num_train_epochs = self.config["epochs"],
			learning_rate=self.config["lr"],
			load_best_model_at_end=True,
			metric_for_best_model = "accuracy",
			greater_is_better= True
			)

		trainer = Trainer(model=self.model.model, 
		args=training_args, 
		train_dataset=train_data, 
		eval_dataset=val_data,
		compute_metrics = self.compute_metrics)
		trainer.train()
		
		# Test

		outputs = trainer.predict(test_data)
		y_pred = outputs.predictions.argmax(1)
		y_true = test_data['labels'] 
		conf_matrix= confusion_matrix(y_true,y_pred).ravel()
		
		return conf_matrix
	
	

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
		results_total = pd.DataFrame(columns=["TP","TN","FP","FN"]) #tn, fp, fn, tp)

		for k in range(self.config['split']):
			train_parts = self.dataset_parts.copy()
			test_data = train_parts.pop(k)
			train_data = concatenate_datasets(train_parts)

			conf_matrix = self.train_test_loop(self.model,self.config['lr'],train_data,test_data)
			results_total.loc[len(results_total)] = conf_matrix
		return results_total

	def run_simple(self):

			data_split = self.data_complete.train_test_split(self.config['split'])
			train_data = data_split['train']
			test_data = data_split['test']
			conf_matrix = self.train_test_loop(self.model,self.config['lr'],train_data,test_data)
			return conf_matrix


	def run_turn(self,requirements):
		self.config = requirements
		self.load_model()
		self.model.create_model()
		self.load_data(self.config["dataset"],self.config["kfold"],self.config["split"])
		# self.model = self.load_model()
		if self.config["kfold"]:
			results = self.run_kfold()
			
		else: 
			results = self.run_simple()
		# self.export_results(results)
		return results




