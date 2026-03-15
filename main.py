#small: 60
#large: 99

import pandas as pd
import numpy as np
import time

def main():
    dataset = input("Type the number of the dataset you want to run." + "\n\n" + "1) Small dataset" + "\n" + "2) large dataset" + "\n" + "3) Sanity Check 1" + "\n" + "4) Sanity Check 2" + "\n")
    
    filename = ""
    
    if dataset == "1":
        filename = "CS170_Small_DataSet__60.txt"
    elif dataset == "2":
        filename = "CS170_Large_DataSet__99.txt"
    elif dataset == "3":
        filename = "SanityCheck_DataSet__1.txt"
    elif dataset == "4":
        filename = "SanityCheckDataSet__2.txt"
    
    algorithm = input("Type the number of the algorithm you want to run." + "\n\n" + "1) Forward Select" + "\n" + "2) Backward Elimination" + "\n")

    start = time.time()

    if algorithm == "1":
        forward_select(filename)
    elif algorithm == "2":
        backward_elimination(filename)
        
    end = time.time()
    
    if algorithm == "1":
        print(f'Time to run forward selection: {round(end - start, 2)} seconds')
    elif algorithm == "2":
        print(f'Time to run backward elimination: {round(end - start, 2)} seconds')


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
            if feature not in iterative_best_features:
                for c in range(0, len(table[:, 0])):
                    if table[nearest_neighbor_classify(table, feature, iterative_best_features, c, True)][0] == table[c][0]:
                        correct_count += 1               

                iterative_best_features_cpy = iterative_best_features.copy()
                iterative_best_features_cpy.append(feature)
                
                accuracy = round((correct_count/len(table[:, 0]))*100, 1)

                print(f'Accuracy for feature(s) {iterative_best_features_cpy}: {accuracy}%')

                level_max_accuracy = max(level_max_accuracy, accuracy)

                if level_max_accuracy == accuracy:
                    best_feature = feature
                
        max_accuracy = max(max_accuracy, level_max_accuracy)
        iterative_best_features.append(best_feature)
        
        with open("iterations.txt", "a", encoding="utf-8") as file:
            file.write(f"Features used: {iterative_best_features} with accuracy: {level_max_accuracy}%\n")
        
        if max_accuracy == level_max_accuracy:
            bestest_features = iterative_best_features.copy()
            
        print(f'\n\nCurrent best features: {bestest_features} with current accuracy: {max_accuracy}% \n\n')
    
    correct_count = 0
    
    for c in range(0, len(table[:, 0])):
        if table[nearest_neighbor_classify_all_features(table, bestest_features, c)][0] == table[c][0]:
            correct_count += 1    
            
    accuracy = round((correct_count/len(table[:, 0]))*100, 1)
    
    print(f'Accuracy for all feature(s) {bestest_features}: {accuracy}% \n\n')
    
    print(f'Best Features: {bestest_features} with Accuracy: {max_accuracy}% \n')


def backward_elimination(filename):
    table = pd.read_csv(filename, sep='\s+', header = None).to_numpy()
    
    bestest_features = list(range(1, len(table[0])))
    
    iterative_best_features = list(range(1, len(table[0])))
    
    iterative_worst_features = []
   
    max_accuracy = 0
    
    print(f'Current features: {bestest_features} \n\n')
    
    correct_count = 0
    
    for c in range(0, len(table[:, 0])):
        if table[nearest_neighbor_classify_all_features(table, bestest_features, c)][0] == table[c][0]:
            correct_count += 1    
            
    accuracy = round((correct_count/len(table[:, 0]))*100, 1)
    
    print(f'Accuracy for all feature(s) {bestest_features}: {accuracy}% \n\n')
    
    for f in range(1, len(table[0])):
        worst_feature = 0
        level_max_accuracy = 0
        # level_min_accuracy = np.inf
        for feature in range(1, len(table[0])):
            correct_count = 0
            if feature not in iterative_worst_features:
                for c in range(0, len(table[:, 0])):
                    if table[nearest_neighbor_classify(table, feature, iterative_best_features, c, False)][0] == table[c][0]:
                        correct_count += 1               

                iterative_best_features_cpy = iterative_best_features.copy()
                iterative_best_features_cpy.remove(feature)
                
                accuracy = round((correct_count/len(table[:, 0]))*100, 1)

                print(f'Accuracy for feature(s) {iterative_best_features_cpy}: {accuracy}%')

                level_max_accuracy = max(level_max_accuracy, accuracy)
                # level_min_accuracy = min(level_min_accuracy, accuracy)

                if level_max_accuracy == accuracy:
                    worst_feature = feature
                
        iterative_worst_features.append(worst_feature)
        iterative_best_features.remove(worst_feature)
        
        max_accuracy = max(max_accuracy, level_max_accuracy)
        if max_accuracy == level_max_accuracy:
            bestest_features = iterative_best_features.copy()
        
        with open("iterations.txt", "a", encoding="utf-8") as file:
            file.write(f"Features after elimination: {iterative_best_features} with accuracy: {level_max_accuracy}%\n")
        
        print(f'\n\nCurrent best features: {bestest_features} with current accuracy: {max_accuracy}% \n\n')
        
        if len(iterative_best_features) == 1:
            break
        
    print(f'Best Features: {bestest_features} with Accuracy: {max_accuracy}%')


def nearest_neighbor_classify(table, feature_num, best_features, curr_index, isForward):
    features_check = best_features.copy()
    if isForward:
        features_check.append(feature_num)
    else:
        features_check.remove(feature_num)
    np_features_check = np.array(features_check)
    
    feature_subset = table[:, np_features_check]
    
    test_row = feature_subset[curr_index]
    
    distances = np.sum((feature_subset - test_row)**2, axis=1)
    distances[curr_index] = np.inf
    
    curr_lowest = np.argmin(distances)
    
    return curr_lowest
    

def nearest_neighbor_classify_all_features(table, features_check, curr_index):
    np_features_check = np.array(features_check)
    
    feature_subset = table[:, np_features_check]
    
    test_row = feature_subset[curr_index]
    
    distances = np.sum((feature_subset - test_row)**2, axis=1)
    distances[curr_index] = np.inf
    
    curr_lowest = np.argmin(distances)
    
    return curr_lowest

main() 