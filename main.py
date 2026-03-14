#small: 60
#large: 99

import pandas as pd
import numpy as np
import math

def main():
    dataset = input("Type the number of the dataset you want to run." + "\n\n" + "1) Small dataset" + "\n" + "2) large dataset" + "\n")
    
    filename = ""
    
    if dataset == "1":
        filename = "CS170_Small_DataSet_60.txt"
    elif dataset == "2":
        filename = "CS170_Large_DataSet_99.txt"
    elif dataset == "3":
        filename = "SanityCheck_DataSet__1.txt"
    elif dataset == "4":
        filename = "SanityCheck_DataSet__2.txt"
    
    algorithm = input("Type the number of the algorithm you want to run." + "\n\n" + "1) Forward Select" + "\n" + "2) Backward Elimination" + "\n")

    if algorithm == "1":
        forward_select(filename)
    # elif algorithm == "2":
    #     backward_elimination(filename)


def forward_select(filename):
    table = pd.read_csv(filename).to_numpy()
    
    print(table)
    
    correct_count = 0
    
    best_features = []
    
    feature_counter = 0
    
    max_accuracy = 0
    
    for feature in range(1, len(table[0]) + 1):
        for c in range(0, len(table[:, 0])):
            if table[nearest_neighbor_classify(table, feature, best_features, table[c][feature])][0] == table[c][0]:
                correct_count += 1
                
        accuracy = correct_count/len(table[:, 0])
        max_accuracy = max(max_accuracy, accuracy)
        
        if max_accuracy == accuracy and len(best_features) == feature_counter + 1:
            best_features[feature_counter - 1] = feature
        elif max_accuracy == accuracy:
            best_features.append(feature)
            feature_counter += 1


def nearest_neighbor_classify(table, feature_num, best_features, curr_feature):
    min_distance = math.inf
    curr_lowest = 0
    for f in range (0, len(table[:, feature_num])):
        distance = np.abs(curr_feature - table[f][feature_num])
        for b in best_features:
            distance += np.abs(curr_feature - table[f][b])
        min_distance = min(min_distance, distance)
        
        if min_distance == distance:
            curr_lowest = f
    
    return curr_lowest
    
    
main() 