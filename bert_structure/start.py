# from bayes_opt import BayesianOptimization
import pandas as pd
import json, datetime, sys,os
import run_model
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


	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
	runnable_model = run_model.RunableModel()
	retults,train_log = runnable_model.run_turn(config)
	export_results(retults,config,train_log)
	




