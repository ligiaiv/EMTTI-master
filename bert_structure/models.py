import torch
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import Trainer


class BERTimbau():
    def __init__(self,parameters):
        self.num_labels = parameters["num_labels"]
    def create_model(self):
        # self.model = AutoModelForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased',num_labels=2,problem_type="multi_label_classification")
        self.model = AutoModelForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased',num_labels=2)
        # self.model.classifier = torch.nn.Linear(self.model.pooler.dense.in_features, self.num_labels)
        
class MultilingualBERT():
    def __init__(self,parameters):
        self.num_labels = parameters["num_labels"]

    def create_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(model='bert-base-multilingual-cased')
    
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