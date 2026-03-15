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
        filename = "SanityCheckDataSet__2.txt"
    
    algorithm = input("Type the number of the algorithm you want to run." + "\n\n" + "1) Forward Select" + "\n" + "2) Backward Elimination" + "\n")

    if algorithm == "1":
        forward_select(filename)
    # elif algorithm == "2":
    #     backward_elimination(filename)


def forward_select(filename):
    table = pd.read_csv(filename, sep='\s+', header = None).to_numpy()
    
    bestest_features = []
    
    iterative_best_features = []
    
    max_accuracy = 0
    
    print(f'Current features: {bestest_features} \n\n')
    
    for f in range(1, len(table[0])):
        best_feature = 0
        level_max_accuracy = 0
        for feature in range(1, len(table[0])):
            correct_count = 0
            # added = False
            if feature not in iterative_best_features:
                for c in range(0, len(table[:, 0])):
                    if table[nearest_neighbor_classify(table, feature, iterative_best_features, c)][0] == table[c][0]:
                        correct_count += 1

                accuracy = correct_count/len(table[:, 0])

                print(f'Accuracy for feature {feature}: {accuracy}')

                level_max_accuracy = max(level_max_accuracy, accuracy)

                if level_max_accuracy == accuracy:
                    best_feature = feature
                
        max_accuracy = max(max_accuracy, level_max_accuracy)
        iterative_best_features.append(best_feature)
        if max_accuracy == level_max_accuracy:
            bestest_features = iterative_best_features.copy()
        print(f'\n\nCurrent best features: {bestest_features} \n\n')
        
    print(f'Best Features: {bestest_features} with Accuracy: {max_accuracy}')


def nearest_neighbor_classify(table, feature_num, best_features, curr_index):
    features_check = best_features.copy()
    features_check.append(feature_num)
    
    np_features_check = np.array(features_check)
    
    feature_subset = table[:, np_features_check]
    
    test_row = feature_subset[curr_index]
    
    distances = np.sum((feature_subset - test_row)**2, axis=1)
    
    distances[curr_index] = np.inf
    
    curr_lowest = np.argmin(distances)
    
    return curr_lowest
    
    
    
    
main() 