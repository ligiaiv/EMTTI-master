# from bayes_opt import BayesianOptimization
import pandas as pd
import json, datetime, sys,os, time
from transformer_run_model import TransformerRunableModel
from torch_run_model import TorchRunableModel
from keras_run_model import KerasRunableModel
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch

print("TORCH VERSION: ",print(torch.__version__))

def export_results(results,config,train_log):
	table = results.to_csv(None)

	config["device"] = str(config["device"])
	output_dict = {"results":table,
				"config":config}
	day = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")
	filename = "results/{}_{}_{}".format(config["dataset"][:5],config["model"],day)
	with open(filename,'w') as outfile:
		json.dump(output_dict,outfile)
	with open(filename+"_log",'w') as outfile:
		json.dump(train_log,outfile)



if len(sys.argv) < 2:
	print(sys.argv)
	print("Please specify the configuration file.")
	quit()

if __name__ == '__main__':
	
	tic = time.process_time()
	device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")

	print("Device used:",device)
	#
	#	CUDA test
	#
	if torch.cuda.is_available():
		print("torch.cuda.is_available()",torch.cuda.is_available())
		print("torch.cuda.device_count()",torch.cuda.device_count())
		print("torch.cuda.current_device()",torch.cuda.current_device())
		print("torch.cuda.get_device_name(0)",torch.cuda.get_device_name(0))

	CONFIG_FILE = sys.argv[1]
	print("This FOLDER",os.getcwd())
	files = [f for f in os.listdir('.') if os.path.isfile(f)]
	# print(files)
	with open("BERT/"+CONFIG_FILE) as infile:
		config = json.load(infile)
	print(config)

	config["device"] = device
	# print(str(device))
	# quit()
	results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
	train_logs = []
	for i in range(config["turns"]):
		if config["base"] == "transformer":
			runnable_model = TransformerRunableModel()
		else:
			runnable_model = KerasRunableModel()
		results,train_log = runnable_model.run_turn(config)
		print(results)
		results_total = results_total.append(results)
		train_logs.append(train_log)
	export_results(results_total,config,train_logs)

	toc = time.process_time()
	print("\nPROCESSING TIME: ",toc-tic)
	




