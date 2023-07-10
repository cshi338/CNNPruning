To first train the resnet18 model, run the following code:
"python3 UnprunedResnet18.py" 
A checkpoint of a resnet18 model that was able to achieve 93% test accuracy will be saved in the /checkpoint directory
Copy output of UnprunedResnet18.py from terminal into a txt file named "pretrainedmodeloutput.txt"

To execute one-shot pruning, run the following code:
"python3 OneShotPrunedModels.py"
Copy output of OneShotPrunedModels.py from terminal into a txt file named "oneshotpruned.txt"

To execute iterative pruning, run the following code:
"python3 IterativePrunedModels.py"
Results will be written to "iterativeOutput.csv"

To generate results, including maximum test set accuracies for each of the different models, as well as graphs showing test set accuracies vs number of epochs, execute the following code:
"python3 read.py"


