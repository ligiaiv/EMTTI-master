import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time


from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments



class BERTimbau():
	def __init__(self,parameters):
		self.num_labels = parameters["num_labels"]
	def create_model(self):
		# self.model = AutoModelForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased',num_labels=2,problem_type="multi_label_classification")
		self.model = AutoModelForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased',num_labels=2)
		# self.model.classifier = torch.nn.Linear(self.model.pooler.dense.in_features, self.num_labels)
		self.tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False,is_split_into_words=True)
	# def train(self,train_data,val_data):
	# 	max_batch_size = 4
	# 	training_args = TrainingArguments(
	# 		# label_names=["label"],
	# 		output_dir = "./results", 
	# 		do_train = True,
	# 		do_eval = True,
	# 		evaluation_strategy = "steps",
	# 		save_strategy = "steps",
	# 		logging_dir='./logs',
	# 		logging_strategy= "steps",
	# 		logging_steps=400,
	# 		eval_steps=400,
	# 		save_steps= 400,
	# 		num_train_epochs = self.config["epochs"],
	# 		learning_rate=self.config["lr"],
	# 		weight_decay=0.01,
	# 		load_best_model_at_end=True,
	# 		metric_for_best_model = "accuracy",
	# 		greater_is_better= True,
	# 		#Performance part
	# 		per_device_train_batch_size=max_batch_size,
	# 		per_device_eval_batch_size=max_batch_size,
	# 		)
	# 	print("TRAINDATA",train_data.features)

	# 	print("VALDATA",val_data.features)
	# 	self.trainer = Trainer(
	# 		model=self.model.model, 
	# 		args=training_args, 
	# 		train_dataset=train_data, 
	# 		eval_dataset=val_data,
	# 		compute_metrics = self.compute_metrics
	# 	)
	# 	print(self.trainer.train_dataset[0])

	# 	self.trainer.train()

	# def test(self,test_data):
	# 	outputs = self.trainer.predict(test_data)
	# 	self.trainer.save_model("./models/model")

class MultilingualBERT():
	def __init__(self,parameters):
		self.num_labels = parameters["num_labels"]

	def create_model(self):
		self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=2)
		self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased',do_lower_case=False,is_split_into_words=True)
	
	# def train(self,train_data,val_data):
	# 	max_batch_size = 4
	# 	training_args = TrainingArguments(
	# 		# label_names=["label"],
	# 		output_dir = "./results", 
	# 		do_train = True,
	# 		do_eval = True,
	# 		evaluation_strategy = "steps",
	# 		save_strategy = "steps",
	# 		logging_dir='./logs',
	# 		logging_strategy= "steps",
	# 		logging_steps=400,
	# 		eval_steps=400,
	# 		save_steps= 400,
	# 		num_train_epochs = self.config["epochs"],
	# 		learning_rate=self.config["lr"],
	# 		weight_decay=0.01,
	# 		load_best_model_at_end=True,
	# 		metric_for_best_model = "accuracy",
	# 		greater_is_better= True,
	# 		#Performance part
	# 		per_device_train_batch_size=max_batch_size,
	# 		per_device_eval_batch_size=max_batch_size,
	# 		)
	# 	print("TRAINDATA",train_data.features)

	# 	print("VALDATA",val_data.features)
	# 	self.trainer = Trainer(
	# 		model=self.model.model, 
	# 		args=training_args, 
	# 		train_dataset=train_data, 
	# 		eval_dataset=val_data,
	# 		compute_metrics = self.compute_metrics
	# 	)
	# 	print(self.trainer.train_dataset[0])

	# 	self.trainer.train()

	# def test(self,test_data):
	# 	outputs = self.trainer.predict(test_data)
	# 	self.trainer.save_model("./models/model")

class LSTMgeneral():
	def __init__(self,parameters):
		self.num_labels = parameters["num_labels"]
		self.parameters = parameters
		self.model_name = "LSTM"
	def create_model(self,vocab_size):
		self.model = LSTMClassifier(embedding_dim = self.parameters["emb_dim"],
									hidden_dim = self.parameters["hidden_dim"],
									vocab_size = vocab_size,
									target_size = self.parameters["num_labels"]
									)
		# self.optimizer = optim.Adam(self.model.parameters(),lr = self.parameters["lr"])
		self.optimizer = optim.RMSprop(self.model.parameters(),lr = self.parameters["lr"])
		# self.optimizer = optim.SGD(self.model.parameters(),lr = self.parameters["lr"])
		self.criterion = nn.CrossEntropyLoss()
		# self.criterion = nn.NLLLoss2d()
		# self.criterion = nn.BCEWithLogitsLoss()
	def train(self,iterator,metric):
		epoch_loss = 0
		epoch_acc = 0

		self.model.train()  # Train mode is on

		for x_batch, y_batch in iterator:
			x = x_batch.type(torch.LongTensor)
			y = y_batch.type(torch.LongTensor)

			if len(y.shape) == 1:
				if self.num_labels==2:
					y_onehot = torch.column_stack((y, y.logical_not()))
				elif self.num_labels==1:
					y_onehot = y[:,None]

			self.optimizer.zero_grad()  # Reset the gradients
			y_pred = self.model(x)
			# y_me = F.softmax(y_pred,dim = 1)
			# my_CE = ((torch.log2(y_pred)*y).sum(axis = 1))*(-1)
			# print("my_CE",my_CE)
			# print(my_CE.mean())
			print(y_pred.shape,y.shape)
			
			loss = self.criterion(y_pred, y)
			print(loss)
			acc = metric(y_pred, y_onehot)

			a = list(self.model.parameters())[0].clone()

			loss.backward()  ## backward propagation / calculate gradients
			self.optimizer.step()  ## update parameters
			b = list(self.model.parameters())[0].clone()
			print("weights equal???",torch.equal(a.data, b.data))
			epoch_loss += loss.item()
			epoch_acc += acc["accuracy"]

			print("In batch: \t loss: {},\r".format(loss))

		return epoch_loss / len(iterator), epoch_acc / len(iterator)

	def evaluate(self, iterator,metric):
		
		epoch_loss = 0
		epoch_acc = 0
		
		self.model.eval() #Evaluation mode is on
		
		# with torch.no_grad():
		
		for x_batch, y_batch  in iterator:
			x = x_batch.type(torch.LongTensor)
			y = y_batch.type(torch.FloatTensor)
			# print("x,y",x,y)
			if len(y.shape) == 1:
				if self.num_labels==2:
					y = torch.column_stack((y, y.logical_not()))
				elif self.num_labels==1:
					y = y[:,None]
			predictions = self.model(x) 
			loss = self.criterion(predictions, y)
			# print("loss",loss)
			acc = metric(predictions, y)
			epoch_loss += loss
			epoch_acc += acc["accuracy"]
				
		return epoch_loss / len(iterator), epoch_acc / len(iterator)

	def predict(self,iterator):
		all_predictions = np.ndarray((0,self.num_labels))
		all_targets = np.ndarray((0,self.num_labels))
		self.model.eval() #Evaluation mode is on
		

		for x_batch, y_batch  in iterator:
			x = x_batch.type(torch.LongTensor)
			y = y_batch.type(torch.FloatTensor)
			# print("x,y",x,y)
			if len(y.shape) == 1:
				if self.num_labels==2:
					y = torch.column_stack((y, y.logical_not()))
				elif self.num_labels==1:
					y = y[:,None]
			predictions = self.model(x) 
			# print("SIZE",predictions.size())
			all_predictions = np.append(all_predictions, predictions, axis=0)
			all_targets = np.append(all_targets, y, axis=0)
			# loss = self.criterion(predictions, y)
			# # print("loss",loss)
			# acc = metric(predictions, y)
		return all_predictions,all_targets

	def save(self, name=None):

		if name is None:
			prefix = './models/' + self.model_name + '_'
			name = time.strftime(prefix + '%m%d_%H_%M_%S.pkl')
		torch.save(self.model.state_dict(),name)
		print("Model saved successfully")
		# t.save(self.state_dict(), name)
		# return name

class MyTrainer(Trainer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def compute_loss(self, model, inputs, return_outputs=False):
		# print("IN MY TRAINER")
		# print("INPUTS",inputs)
		"""
		How the loss is computed by Trainer. By default, all models return the loss in the first element.

		Subclass and override for custom behavior.
		"""
		if self.label_smoother is not None and "labels" in inputs:
			labels = inputs.pop("labels")
		else:
			labels = None
		outputs = model(**inputs)
		# print("OUTPUTS",outputs)
		# print("LABELS",labels)
		# Save past state if it exists
		# TODO: this needs to be fixed and made cleaner later.
		if self.args.past_index >= 0:
			self._past = outputs[self.args.past_index]

		if labels is not None:
			loss = self.label_smoother(outputs, labels)
		else:
			# We don't use .loss here since the model may return tuples instead of ModelOutput.
			loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

		return (loss, outputs) if return_outputs else loss



class LSTMClassifier(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size, lstm_layers = 1):
		super(LSTMClassifier,self).__init__()

		self.lstm_layers = lstm_layers
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
		self.dropout = nn.Dropout(0.25)
		# self.dropout2 = nn.Dropout(0.5)

		# self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
		self.lstm = nn.LSTM(input_size = embedding_dim,hidden_size = hidden_dim,num_layers=self.lstm_layers, batch_first=True)

		self.linear = nn.Linear(hidden_dim,target_size)

	def forward(self,sentence):
		# print("sentence in forward",sentence)
		# Hidden and cell state definion
		# print(type(sentence))



		##Sentences arrive here as tensors of ints
		h0 = torch.zeros((self.lstm_layers, sentence.size(0), self.hidden_dim))
		c0 = torch.zeros((self.lstm_layers, sentence.size(0), self.hidden_dim))
		
		# # Initialization fo hidden and cell states
		torch.nn.init.xavier_normal_(h0)
		torch.nn.init.xavier_normal_(c0)

		embeds = self.word_embeddings(sentence)
		# print("embeds",embeds.size())
		# print("embeds",embeds.size())

		lstm_out, (hn, cn) = self.lstm(embeds,(h0,c0))
		drop = self.dropout(lstm_out)
		# lstm_out, (hn, cn) = self.lstm(embeds)
		# print("lstm_out",lstm_out.size())

		# out, (hidden, cell) = self.lstm(out, (h,c))
		# out = self.dropout(out)
		# # The last hidden state is taken
		# out = torch.relu_(self.fc1(out[:,-1,:]))
		# out = self.dropout(out)
		# out = torch.sigmoid(self.fc2(out))


		linear_out = self.linear(drop[-1,:,:])
		# print("linear_out",linear_out.size())

		# scores = torch.sigmoid(linear_out)
		# scores = F.softmax(linear_out,dim=-1)
		# print("scores",scores.size())
		scores = linear_out
		return scores