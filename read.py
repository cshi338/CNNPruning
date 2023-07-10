import csv
import numpy as np
import matplotlib.pyplot as plt
sparsityRatios = [0.5, 0.75, 0.9]


resnet18 = []
with open("pretrainedmodeloutput.txt") as file:
    lines = file.readlines()
    for line in lines:
        if "Test Acc" in line:
            temp = line.split("Test Acc: ",1)[1]
            temp = temp.split('%')[0]
            resnet18.append(float(temp))
plt.ylabel('Test Set Accuracy')
plt.xlabel('Epoch Number')
plt.title('Test Set Accuracy vs Epoch of Untrained resnet18 Model')
plt.plot(resnet18)
plt.show()
print("Maximum Test Set Accuarcy of resnet18 Model: " + str(max(resnet18)))
print()

oneshot = []
maximums = []
with open("oneshotpruned.txt") as file:
    lines = file.readlines()
    iteration = []
    for line in lines:
        if "Test Acc" in line:
            temp = line.split("Test Acc: ",1)[1]
            temp = temp.split('%')[0]
            iteration.append(float(temp))
            if len(iteration) == 90:
                oneshot.append(iteration)
                iteration = []
for x in range(0,len(oneshot)):
    plt.ylabel('Test Set Accuracy')
    plt.xlabel('Epoch Number')
    plt.title('Test Set Accuracy vs Epoch of One Shot Pruned Model')
    maximums.append(max(oneshot[x]))
    plt.plot(oneshot[x], label = "Sparsity Ratio = " + str(sparsityRatios[x]))
plt.legend()
plt.show()
print("Maximum Test Set Accuracies for One Shot Pruning: " + str(maximums))
print()

iterative = []
with open("iterativeOutput.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        iterative.append(",".join(line).replace(",",""))

x = 0
output = []
iteration = []
globalSparsity = []
maximums = []
for line in iterative:
    if "Test Acc" in line:
        temp = line.split("Test Acc: ",1)[1]
        temp = temp.split('%')[0]
        iteration.append(float(temp))
        if len(iteration) == 90:
            output.append(iteration)
            iteration = []
    if "Global sparsity" in line:
        temp = line.split("Global sparsity: ",1)[1]
        temp = temp.split('%')[0]
        if float(temp)/100 >= sparsityRatios[x]:
            globalSparsity.append(float(temp))
            for y in range(len(output)):
                plt.ylabel('Test Set Accuracy')
                plt.xlabel('Epoch Number')
                plt.title('Test Set Accuracy vs Epoch of Iterative Pruned Model with Sparsity Ratio = ' + str(sparsityRatios[x]))
                maximums.append(max(output[y]))
                plt.plot(output[y], label = "Global Sparisty = " + str(globalSparsity[y]))
            plt.legend()
            plt.show()
            print()
            iteration = []
            output = []
            globalSparsity = []
            print("Maximum Test Set Accuracies for " + str(sparsityRatios[x]) + " Iterative Pruning: " + str(maximums))
            maximums = []
            x += 1
        else:
            globalSparsity.append(float(temp))
