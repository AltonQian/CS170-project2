#small: 60
#large: 99

import pandas as pd
import numpy as np
import time

def main():
    #gui for the project
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

    #start the timer to time the algorithm
    start = time.time()

    if algorithm == "1":
        forward_select(filename)
    elif algorithm == "2":
        backward_elimination(filename)
    
    #end the timer 
    end = time.time()
    
    if algorithm == "1":
        print(f'Time to run forward selection: {round(end - start, 2)} seconds')
    elif algorithm == "2":
        print(f'Time to run backward elimination: {round(end - start, 2)} seconds')


def forward_select(filename):
    #utilized pandas to read the file, and tabulate it using the spaces as seperators (no headers). I them converted it into a numpy array
    table = pd.read_csv(filename, sep='\s+', header = None).to_numpy()
    
    #holds the global best features that yield the highest accuracy 
    bestest_features = []
    
    #holds the local best features that yield the highest accuracy at every iteration of determining which feature to add. at the end, it should have every feature.
    iterative_best_features = []
    
    max_accuracy = 0
    
    print(f'Current features: {bestest_features} \n\n')
    
    #first loop through the features (columns). This loop is to make sure every feature is added to the "used" set
    for f in range(1, len(table[0])):
        best_feature = 0
        level_max_accuracy = 0
        #nested for loop. loops through the features to check which features when added to the iterative_best_features set yields the higher accuracy
        for feature in range(1, len(table[0])):
            correct_count = 0
            #makes sure that the feature hasn't already been considered
            if feature not in iterative_best_features:
                #loops through every row (every data entry)
                for c in range(0, len(table[:, 0])):
                    #calls on nearest_neighbor_classify helper function, and if it matches the current row's class, the correct_count goes up by 1
                    if table[nearest_neighbor_classify(table, feature, iterative_best_features, c, True)][0] == table[c][0]:
                        correct_count += 1               

                #make a copy of the iterative_best_features to append the current feature. copy because we don't know yet if this feature is the highest accuracy 
                #at this level. this is just for printing purposes
                iterative_best_features_cpy = iterative_best_features.copy()
                iterative_best_features_cpy.append(feature)
                
                #calculate accuracy
                accuracy = round((correct_count/len(table[:, 0]))*100, 1)

                print(f'Accuracy for feature(s) {iterative_best_features_cpy}: {accuracy}%')

                #updates the highest accuracy at this level, and then update the best_feature if the current feature has a higher accuracy
                level_max_accuracy = max(level_max_accuracy, accuracy)

                if level_max_accuracy == accuracy:
                    best_feature = feature
        
        #updates the global max accuracy to see if a better subset of features has been found
        max_accuracy = max(max_accuracy, level_max_accuracy)
        iterative_best_features.append(best_feature)
        
        #write the new set (iterative_best_features) with its accuracy to a file so I can refer to it for analysis
        with open("iterations_forward.txt", "a", encoding="utf-8") as file:
            file.write(f"Features used: {iterative_best_features} with accuracy: {level_max_accuracy}%\n")
        
        #updates the global best features if the global highest accuracy gets changed
        if max_accuracy == level_max_accuracy:
            bestest_features = iterative_best_features.copy()
            
        print(f'\n\nCurrent best features: {bestest_features} with current accuracy: {max_accuracy}% \n\n')
    
    #lines 102 - 113 basically does one last iteration of every row (data entry) but this time including every feature.
    correct_count = 0
    
    for c in range(0, len(table[:, 0])):
        if table[nearest_neighbor_classify_all_features(table, list(range(1, len(table[0]))), c)][0] == table[c][0]:
            correct_count += 1    
            
    accuracy = round((correct_count/len(table[:, 0]))*100, 1)
    
    print(f'Accuracy for all feature(s) {list(range(1, len(table[0])))}: {accuracy}% \n\n')
    
    with open("iterations_forward.txt", "a", encoding="utf-8") as file:
        file.write(f"All features: {list(range(1, len(table[0])))} with accuracy: {accuracy}%\n")
    
    print(f'Best Features: {bestest_features} with Accuracy: {max_accuracy}% \n')


def backward_elimination(filename):
    table = pd.read_csv(filename, sep='\s+', header = None).to_numpy()
    
    #set the flobal best feature and iterative best feature set to every feature
    bestest_features = list(range(1, len(table[0])))
    
    iterative_best_features = list(range(1, len(table[0])))
    
    iterative_worst_features = []
    
    print(f'Current features: {bestest_features} \n\n')
    
    #lines 131 - 144 basically does one iteration of every row (data entry) to include every feature before it starts removing.
    correct_count = 0
    
    for c in range(0, len(table[:, 0])):
        if table[nearest_neighbor_classify_all_features(table, bestest_features, c)][0] == table[c][0]:
            correct_count += 1    
            
    accuracy = round((correct_count/len(table[:, 0]))*100, 1)
    
    print(f'Accuracy for all feature(s) {bestest_features}: {accuracy}% \n')
    max_accuracy = accuracy
    print(f'\n\nCurrent best features: {bestest_features} with current accuracy: {max_accuracy}% \n\n')
    
    with open("iterations.txt", "a", encoding="utf-8") as file:
        file.write(f"All features: {iterative_best_features} with accuracy: {accuracy}%\n")
    
    #similar logic as forward selection
    for f in range(1, len(table[0])):
        worst_feature = 0
        level_max_accuracy = 0
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

                if level_max_accuracy == accuracy:
                    worst_feature = feature
        
        #add the worst feature found to the the "already considered" set, and remove the worst feature from the iterative_best_features set
        iterative_worst_features.append(worst_feature)
        iterative_best_features.remove(worst_feature)
        
        max_accuracy = max(max_accuracy, level_max_accuracy)
        if max_accuracy == level_max_accuracy:
            bestest_features = iterative_best_features.copy()
        
        with open("iterations.txt", "a", encoding="utf-8") as file:
            file.write(f"Features after elimination: {iterative_best_features} with accuracy: {level_max_accuracy}%\n")
        
        print(f'\n\nCurrent best features: {bestest_features} with current accuracy: {max_accuracy}% \n\n')
        
        #stop the loop once the last feature has already been considered, as it's pointless to calculate a zero feature set
        if len(iterative_best_features) == 1:
            break
        
    print(f'Best Features: {bestest_features} with Accuracy: {max_accuracy}%')


#helper function to calculate the nearest neighbor
def nearest_neighbor_classify(table, feature_num, best_features, curr_index, isForward):
    #make a copy of the passed in best_feature array to not mess with the passed in array
    features_check = best_features.copy()
    
    #if it's a forward selection algorithm, add the current feature to the current set of best features to find the nearest neighbor
    #of adding that feature. if it's a backward elimination algorithm, remove the current feature to the find the nearest neighbor of
    #removing that feature
    if isForward:
        features_check.append(feature_num)
    else:
        features_check.remove(feature_num)
        
    #turn features_check into a numpy array
    np_features_check = np.array(features_check)
    
    #isolate the columns inside np_features_check (I only want the features inside np_features_check)
    feature_subset = table[:, np_features_check]
    
    #extract the row I'm testing
    test_row = feature_subset[curr_index]
    
    #numpy operation to calculate Euclidean Distance (i didn't add the square root because it's pointless in this case) between the
    #test row's features and every other row's features
    distances = np.sum((feature_subset - test_row)**2, axis=1)
    
    #set the test row distance to infinity because this one doesn't count (since the distance between the test row and itself is just 0)
    distances[curr_index] = np.inf
    
    #find the index of the shortest/smallest distance
    curr_lowest = np.argmin(distances)
    
    return curr_lowest
    
    
#helper function to classify when all the features are being considered (no need to append or remove from features_check).
#everything else is the same as the previous helper funciton.
def nearest_neighbor_classify_all_features(table, features_check, curr_index):
    np_features_check = np.array(features_check)
    
    feature_subset = table[:, np_features_check]
    
    test_row = feature_subset[curr_index]
    
    distances = np.sum((feature_subset - test_row)**2, axis=1)
    distances[curr_index] = np.inf
    
    curr_lowest = np.argmin(distances)
    
    return curr_lowest

main() 