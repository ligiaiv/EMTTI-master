from transformers import AutoModelForSequenceClassification

class BERTimbau():
    def __init__(self,parameters):
        self.num_labels = parameters["num_labels"]
        pass
    def create_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased',num_labels = self.num_labels)
        
