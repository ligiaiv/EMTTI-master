# from bayes_opt import BayesianOptimization
import pandas as pd
import json, datetime, sys,os
import run_model


def export_results(results,config):
	table = results.to_csv(None)

	output_dict = {"results":table,
				"config":config}
	day = datetime.datetime.now().strftime("%d-%m-%y__%H-%M")
	filename = "results/{}_{}_{}_{}".format(config["dataset"][:5],config["model"],day)
	with open(filename,'w') as outfile:
		json.dump(output_dict,outfile)


if len(sys.argv) < 2:
	print(sys.argv)
	print("Please specify the configuration file.")
	quit()

if __name__ == '__main__':


	CONFIG_FILE = sys.argv[1]
	print("This FOLDER",os.getcwd())
	files = [f for f in os.listdir('.') if os.path.isfile(f)]
	print(files)
	with open("BERT/"+CONFIG_FILE) as infile:
		config = json.load(infile)
	print(config)

	runnable_model = run_model.RunableModel()
	retults = runnable_model.run_turn(config)
	export_results(retults,config)
	




