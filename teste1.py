from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForPreTraining
from transformers import AutoModel
from transformers import AdamW
# from torch.utils.data import Dataset,Dataloader
from sklearn.model_selection import train_test_split  
import torch
import helper
from tqdm.auto import tqdm
from transformers import Trainer

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

model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)


data = helper.DataHandler()
data.readDataFile("fakebr.csv")

def tokenize_function(examples):
	return tokenizer(examples, padding="max_length", truncation=True)

tokenized_data = data.map(tokenize_function)
train_data,test_data = tokenized_data.split(0.33)
train_data,val_data = train_data.split(0.1)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(test_data, batch_size=8)

optimizer = AdamW(model.parameters(), lr=5e-5)


from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
model.to(device)

print(train_data)


progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
# print(type(inputs))