# from bayes_opt import BayesianOptimization
import pandas as pd
import json, datetime, sys,os, time
import run_model

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch


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
	CONFIG_FILE = sys.argv[1]
	print("This FOLDER",os.getcwd())
	files = [f for f in os.listdir('.') if os.path.isfile(f)]
	print(files)
	with open("BERT/"+CONFIG_FILE) as infile:
		config = json.load(infile)
	print(config)

	config["device"] = device
	# print(str(device))
	# quit()
	results_total = pd.DataFrame(columns=["TN","FP","FN","TP"]) #tn, fp, fn, tp)
	train_logs = []
	for i in range(config["turns"]):
		runnable_model = run_model.RunableModel()
		results,train_log = runnable_model.run_turn(config)
		print(results)
		results_total = results_total.append(results)
		train_logs.append(train_log)
	export_results(results_total,config,train_logs)

	toc = time.process_time()
	print("\nPROCESSING TIME: ",toc-tic)
	




