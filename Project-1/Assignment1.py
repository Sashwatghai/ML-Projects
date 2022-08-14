"""
    @Author Surajit Kundu
    @Email surajit.113125@gmail.com
    @Roll No 21MM91R09
    @Description: Building a decision tree using ID3 algorithm, Using information gain, and gini index measure 
                for the selection of node of the tree.   
"""
## import the required libraries
import numpy as np
import math
## Define a class
class ID3DecisionTreeClassifier:
    """
        Creating a parameterized constructer, accept the traing dataset as an argument. 
        The tree depth, measure criterion, and the threshold for minimum information gain can be set during the instance creation.
    """
    def __init__(self, training_data, max_depth=-1, criterion='entropy', min_infogain=-1):
        self.dataset = training_data
        self.depth = max_depth
        self.criteria = criterion
        self.node_samples = {}
        self.parent = ""
        self.total_node = 0
        self.min_infogain = min_infogain
    """
        This function return the training dataset
        @Return: 
            self.dataset -> training dataset
    """   
    def training_data(self):
        return self.dataset
    
    """
        This function calculate the gini index of an attribute from the feature set
        @Input: 
            training_data -> training_data used to find the gini index based on their class sample probability
            input_feature_node -> feature attribute from the feature list, on which we want to find gini index
            target_node -> target attribute name
        @Return:
            gini_index -> Gini index of the input feature attribute        
    """
    def gini(self, training_data, input_feature_node, target_node):
        input_attribute_values, input_no_of_elements = np.unique(training_data[input_feature_node], return_counts=True)
        gini_index = 0.0
        for i in range(len(input_attribute_values)):
            class_wise_entropy = training_data.where(training_data[input_feature_node]==input_attribute_values[i]).dropna()
            target_attribute_values, target_no_of_elements = np.unique(class_wise_entropy[target_node], return_counts=True)
            gini_attr_sum = 0.0
            for j in range(len(target_attribute_values)):
                ## Take the sum squre of the probability of target class occurance
                gini_attr_sum += pow((target_no_of_elements[j]/input_no_of_elements[i]),2)
            gini_index += (1-gini_attr_sum) * (input_no_of_elements[i]/np.sum(input_no_of_elements))
        return gini_index    
        
    """
        This function is used to measure the entropy of an attribute from the feature set.
        @Input: 
            node -> Attribute name 
        @Return: 
            node_entropy -> Entropy of the attribute
    """ 
    def entropy(self, node):
        ## Initilize the entopy value as 0.0
        node_entropy = 0.0        
        ## Find the no of distinct class and number of elements in each class
        attribute_values, no_of_elements = np.unique(node, return_counts=True)
        ## Calculate the entopy for each class and sum
        for i in range(len(attribute_values)):
            ## probality of the occurance of each class (positive and negative)
            prob = no_of_elements[i]/np.sum(no_of_elements)
            ## probability multiply with the minus log (base 2) and sum them 
            node_entropy += np.sum(-prob * math.log(prob, 2))
        return node_entropy   
    """
        This function is used to calculate the information gain an attribute from the feature set.
        @Input: 
            training_data -> training dataset
            input_feature_node -> feature attribute from the feature list, on which we want to find information gain
            target_node -> target attribute name
        @Return:
            gain -> Information gain of the input feature attribute
    """
    def InformationGain(self, training_data, input_feature_node, target_node):
        ## Initilize the input_feature_entropy value as 0.0
        input_feature_entropy = 0.0
        target_node_entropy = self.entropy(training_data[target_node])
        attribute_values, no_of_elements = np.unique(training_data[input_feature_node], return_counts=True)
        for i in range(len(attribute_values)):
            class_wise_entropy = self.entropy(training_data.where(training_data[input_feature_node]==attribute_values[i]).dropna()[target_node])
            prob = no_of_elements[i]/np.sum(no_of_elements)
            input_feature_entropy += np.sum(prob * class_wise_entropy)
        gain = target_node_entropy - input_feature_entropy
        return gain   
    
    """
        This function is used to find the best feature attribute based on the maximum information gain
        @Input:
            information_gain -> An array contains information gain of all feature attributes
            feature_nodes -> input feature attributes 
        @Return:
            best_feature_node -> Feature attribute has maximum gain
            feature_nodes -> All input feature aattribues except best feature attribute
            np.max(information_gain) -> maximum information gain
    """
    def BestFeatureAttribute(self, information_gain, feature_nodes):
        ## Index value which has maximum information gain
        max_information_gain_index = np.argmax(information_gain)
        ## find the attribute name from the feature set which has maximum information gain
        best_feature_node = feature_nodes[max_information_gain_index]
        ## Remove the best feature node from the feature list
        feature_nodes = feature_nodes.delete(max_information_gain_index) 
        #print("Max gain ", max_information_gain_index, np.max(information_gain))        
        return best_feature_node, feature_nodes, np.max(information_gain)

    """
        This function build the decision tree using ID3 algorithm
        @Input:
            examples -> Input training data
            target_node -> Target attribute 
            feature_nodes -> Input feature set
    """
    def DecisionTree(self, examples, target_node, feature_nodes, tree_depth=0):
        training_data = self.training_data()
        ## No of distinct class in the target attribute
        no_of_traget_classes = np.unique(examples[target_node])
        ## If all Examples belongs to a single class (positive or negative), return the single-node tree Root
        if len(no_of_traget_classes) == 1:
            return no_of_traget_classes[0]     
        ## If Attributes is empty, return the single-node tree Root, with label = most common value of Targetattribute in Examples
        elif len(feature_nodes) == 0:
            no_of_elements_target_class = sorted(np.unique(training_data[target_node], return_counts=True)[1])
            majority_class_index = np.argmax(no_of_elements_target_class)
            parent_node_class = no_of_traget_classes[majority_class_index]
            return parent_node_class
        else:
            information_gain = [self.gini(examples, feature, target_node) for feature in feature_nodes] if self.criteria == 'gini' else [self.InformationGain(examples, feature, target_node) for feature in feature_nodes]
            #print(information_gain)
            ## best feature attribute using information gain and updated feature nodes excluding the best feature attribute
            best_feature_node, feature_nodes, max_gain = self.BestFeatureAttribute(information_gain, feature_nodes)
            decision_tree = {best_feature_node: {}}
            decision_tree_value_distribution = {best_feature_node: {}}
            best_feature_node_unique_values = np.unique(examples[best_feature_node])
            tree_value_distribution = []
            tree_depth += 1
            if(tree_depth>=self.depth and self.depth>0):
                return decision_tree              
            for value in best_feature_node_unique_values:
                class_wise_data = examples.where(examples[best_feature_node] == value).dropna()
                if max_gain>self.min_infogain:
                    subtree = self.DecisionTree(class_wise_data, target_node, feature_nodes, tree_depth)
                    decision_tree[best_feature_node][value] = subtree
                tree_value_distribution.append(np.array([value,len(class_wise_data)]))
                sk = best_feature_node+" -> "+self.parent+" -> "+str(value)
                self.node_samples[sk] = np.array([value,len(class_wise_data)])
                self.parent = best_feature_node
                self.total_node += 1
            return decision_tree   
        
    """
        This function is used to fit the training data in the decision tree model
        @Input:
            feature_nodes -> Input feature set
            target_node -> Target attribute 
    """
    def fit(self, feature_nodes, target_node):
        training_data = self.training_data()
        self.tree = self.DecisionTree(training_data, target_node, feature_nodes)

    """
        This function predict the target value based on the input test data
        @Input:
            test_data -> input dataset for testing the model
        @Return:
            predicted_values -> Predicted values based on the input data
    """    
    def predict(self, test_data):
        predicted_values = []
        for sample in test_data.to_dict(orient='records'):
             predicted_values.append(1) if self.predict_model(sample, self.tree) is None else predicted_values.append(self.predict_model(sample, self.tree))
        return predicted_values

    """
        This function find the target value for each input sample from the trained tree
        @Input:
            sample -> A single row from the input test data (each feature contains one sample)
            tree -> The decision tree after trained
        @Return:
            pred -> target value for each sample
    """
    def predict_model(self, sample, tree):
        for node in list(sample.keys()):
            if node in list(tree.keys()):
                try:
                    pred = tree[node][sample[node]]
                except:
                    return 1
                if isinstance(pred, dict):
                    return self.predict_model(sample, pred)
                else:
                    return pred   


## import required library for data preprocessing and accuracy measure

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
"""
Section 1: Data read and preprocessing
"""
## Read the dataset
data = pd.read_csv("parkinsons.csv")
## remove the status and name column from the dataset
X = data.drop(['status', 'name'], axis=1)
## keep the target value in y
y = data['status']
features = X.columns
"""
Preprocessing the dataset for building the decision tree
Case 1: MinMax scalling then comparing with the mean and converting to 0, 1.
Case 2: Using KBinsDiscritizer from sklearn for converting the numerical values into intervals
"""
## Case1: Convert the numerical value to boolean
for i in X.columns: 
    X[i] = (X[i] - min(X[i])) / (max(X[i]) - min(X[i]))
    for j in range(len(X[i])):
        #print(X[i][j], X[i].mean())
        if((X[i][j] > X[i].mean())):
            X[i][j] = 1
        else:
            X[i][j] = 0
X = X.dropna().apply(np.int64)    

## Case2: Bin continuous data into intervals.
# from sklearn.preprocessing import KBinsDiscretizer
# est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
# X = est.fit_transform(X)
# X = pd.DataFrame(X,columns=features)


##Train test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

"""
Section 2: Build tree and accuracy measure using entropy. Solution of question 1.
"""
## Accuracy using entropy impurity measure
X_data_e = X_train.join(y_train).dropna().apply(np.int64)
model = ID3DecisionTreeClassifier(X_data_e, criterion='entropy')
model.fit(X_train.columns, y_train.name)
y_pred_e = model.predict(X_test)
print("\nAccuracy score using entropy Test:", accuracy_score(y_test, y_pred_e), " Train:",accuracy_score(y_train, model.predict(X_train))) 

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp=confusion_matrix(y_test, y_pred_e).ravel()
print("TP TN FP FN \n",tp, tn, fp, fn)
print("Tree using entropy: \n\n", model.tree)

"""
Section 3: Build tree and accuracy measure using gini. Solution of question 1.
"""
## Accuracy using gini impurity measure
X_data_g = X_train.join(y_train).dropna().apply(np.int64)
model = ID3DecisionTreeClassifier(X_data_g, criterion='gini')
model.fit(X_train.columns, y_train.name)
y_pred_g = model.predict(X_test)
print("\n\nAccuracy score using gini Test:", accuracy_score(y_test, y_pred_g), " Train:",accuracy_score(y_train, model.predict(X_train))) 

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp=confusion_matrix(y_test, y_pred_g).ravel()
print("TP TN FP FN \n",tp, tn, fp, fn)
print("Tree using gini: \n\n", model.tree)

## Solution of question 1: Analyzing the impact of using individual impurity measures on the prediction.
best_criterion = 'entropy' if accuracy_score(y_test, y_pred_e) > accuracy_score(y_test, y_pred_g) else 'gini'
print("\nSol 1: \n----------------------------------------------------------------\n")
print("Accuracy is high, when we use", best_criterion, "impurity measure\n----------------------------------------------------------------\n")

"""
Section 4: Providing the accuracy by averaging over 10 random 80/20 splits. also print the tree provides best test accuracy. Solution of question 2.
"""
# the accuracy by averaging over 10 random 80/20 splits
dataset = X.join(y).dropna().apply(np.int64)
avg10accuracy = 0.0
all_tree = []
prev_test_accuracy = 0.0
avg_all_accuracy = []
best_X_train, best_y_train, best_X_test, best_y_test = [], [], [], []
best_test_accuracy = 0.0
best_tree = ""
for i in range(10):
    msk = np.random.rand(len(dataset)) < 0.8
    train = dataset[msk]
    test = dataset[~msk]   
    X_train = train.drop(['status'], axis=1)
    y_train = train['status']
    X_test = test.drop(['status'], axis=1)
    y_test = test['status']
    X_data = X_train.join(y_train).dropna().apply(np.int64)
    model = ID3DecisionTreeClassifier(X_data, criterion='entropy')
    model.fit(X_train.columns, y_train.name)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_X_train, best_y_train, best_X_test, best_y_test = X_train, y_train, X_test, y_test
        best_tree = model.tree
    avg_all_accuracy.append(test_accuracy)
    all_tree.append(model.tree)
    prev_test_accuracy = test_accuracy
    avg10accuracy += test_accuracy
print("\nSol 2:\n---------------------------------------------------------------------\n")
print("Accuracy score on averaging over 10 random 80/20 splits is", avg10accuracy/10)
print("\nTree having best accuracy:", best_test_accuracy, "\n\nTree: \n", all_tree[np.argmax(avg_all_accuracy)])  
print("\n---------------------------------------------------------------------\n")

"""
Section 5: Finding the best possible depth limit to be used for your dataset. Providing
a plot explaining the same. Also providing a plot of the test accuracy vs. the total number of nodes in the trees. Solution of question 3.
"""
## Plotting the accuracy over tree depth and number of nodes
X_data_p = best_X_train.join(best_y_train).dropna().apply(np.int64)
train_accuracy = []
test_accuracy = []
depth = []
total_node = []
best_test_acc = 0.0
best_depth_tree = ""
# best_depth = 1
for j in range(1,len(X.columns)):
    model = ID3DecisionTreeClassifier(X_data_p, max_depth=j, criterion='entropy')
    model.fit(best_X_train.columns, best_y_train.name)
    y_pred = model.predict(best_X_test)
    train_acc = accuracy_score(best_y_train, model.predict(best_X_train))
    test_acc = accuracy_score(best_y_test, y_pred)
    if(test_acc>best_test_acc):
        best_depth_tree = model.tree
        best_test_acc = test_acc    
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)  
    depth.append(j)
    total_node.append(model.total_node)
print("\n\nSol 3 & 5:\n-----------------------------------------------------------------\n")
print("Best possible depth: ", depth[np.argmax(test_accuracy)], "Accuracy:",max(test_accuracy))    
print("\nTree at depth", depth[np.argmax(test_accuracy)], "is\n", best_depth_tree)
print("\n-----------------------------------------------------------------\n")
plt.plot(depth, train_accuracy, 'y', label='Train',drawstyle="steps-post", marker='o')
plt.plot(depth, test_accuracy, 'b', label='Test',drawstyle="steps-post", marker='o')
plt.title('Depth vs Accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('DepthVSAccuracy.png')
plt.show()  


plt.plot(total_node, train_accuracy, 'y', label="Train",drawstyle="steps-post", marker='o')
plt.plot(total_node, test_accuracy, 'b', label="Test",drawstyle="steps-post", marker='o')
plt.title('Total Node vs Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Total Node')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('AccuracyVSTotal_Node.png')
plt.show()
print("\n\nPloted Graph are saved in the directory as png images.\n")

"""
Section 6: Performing the pruning operation over the tree with the highest test accuracy and print the final decision tree. 
In question 2 when we get best accuracy, we save the train, test data and train the model using same and prune the the tree to get the final tree with best accuracy 
Solution of question 4 and 5.
"""
# pruning the tree
print("Tree pruining is being performed...\n")
impurities = [0.005, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.30, 0.50]
final_tree = ""
accuracy = 0.0
prun_accuracy = []
for i in impurities:
    model = ID3DecisionTreeClassifier(X_data_p, min_infogain=i, criterion='entropy')
    model.fit(best_X_train.columns, best_y_train.name)
    y_pred = model.predict(best_X_test)
    train_accuracy = accuracy_score(best_y_train, model.predict(best_X_train))
    test_accuracy = accuracy_score(best_y_test, y_pred)
    prun_accuracy.append(test_accuracy)
    if(test_accuracy>=accuracy):
        final_tree = model.tree
        accuracy = test_accuracy
    print(i, "=> Accuracy Test=>", test_accuracy, "Train=>", train_accuracy)
plt.plot(impurities, prun_accuracy, 'b', drawstyle="steps-post", marker='o')
plt.title('Impurities vs Accuracy')
plt.xlabel('Impurities')
plt.ylabel('Accuracy')
plt.legend()
plt.show()      
print("\n\nSol 4 & 5:\n------------------------------------------------------------------\n")    
print("Pruned Tree with highest test accuracy: \n", final_tree, "\n\nTest Accuracy: ", accuracy)
print("\n-------------------------------Thank you-----------------------------------\n")