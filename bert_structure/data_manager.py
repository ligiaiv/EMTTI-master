from datasets import load_dataset, Value,concatenate_datasets

def transformer_load_data(se,datafile,mode,split):
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

def torch_load_data():
	if mode == "kfold":
        pass
	elif mode == "train_test":
        pass
    else: