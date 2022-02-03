# from torch.utils.data import DataLoader
from transformers import AutoTokenizer, training_args
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel
# from transformers import AdamW
# from torch.utils.data import Dataset,Dataloader
from sklearn.model_selection import train_test_split  
import torch
import helper
from tqdm.auto import tqdm
from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_dataset, Value
import numpy as np
#
#   Verify if GPU
#

import tensorflow as tf


print("TESTE2.py")

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
	print('GPU device not found')
else:
  	print('Found GPU at: {}'.format(device_name))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)

model = AutoModelForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased',num_labels = 1)
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False,is_split_into_words=True)

print("\nMODEL EXISTS\n", not(model==None))
# data = helper.DataHandler()
# data = load_dataset()
data = load_dataset('csv',data_files = "Datasets/fakebr.csv")['train']


new_features = data.features.copy()
new_features['labels'] = Value('float')
# data['train']['labels'] = np.array(data['train']['labels'], dtype=np.float32)
data = data.cast(new_features)
print(data.features)
def tokenize_function(examples):
	return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200)

tokenized_data = data.map(tokenize_function,batched=False)
print("LABELS",tokenized_data['labels'])
# quit()
# print(tokenized_data.sentences)
# print(len(tokenized_data[0]['text']['input_ids']))
split_data = tokenized_data.train_test_split(0.33)
train_data = split_data['train']
test_data = split_data['test']
split_data = train_data.train_test_split(0.1)
train_data = split_data['train']
val_data = split_data['test']



from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test_trainer")

trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data)
# print(type(inputs))

trainer.train()