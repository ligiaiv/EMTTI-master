#
#   Script to do necessary preparations
#

from transformers import AutoTokenizer
from transformers import AutoModelForPreTraining
from transformers import AutoModel

#   Downloading BERT models

model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
model = AutoModel.from_pretrained(model='bert-base-multilingual-cased')
tokenizer = AutoTokenizer.from_pretrained(model='bert-base-multilingual-cased',do_lower_case=False)