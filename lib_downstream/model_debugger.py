# put the code of drawing figures here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import config
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from parameter_parser import parameter_parser
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import _tree
import seaborn as sns

class ModelDebugger:
    def __init__(self, model, data = None, features = [], target = ''):
        self.logger = logging.getLogger('PlotUtils')
        self.model = model
        self.data = data
        self.features = features
        self.target = target

    def to_list(self, variable):
        if not isinstance(variable, list):
            return [variable]
        return variable

    def plotDT(self):
        # Plot the decision tree
        plt.figure(figsize=(20, 10))  # Set the size of the plot
        plot_tree(self.model, 
                feature_names=self.features, 
                class_names=self.to_list(self.target),
                filled=True, rounded=True)
        plt.show()

    def print_decision_tree_rules(self):
        tree_ = self.model.tree_
        feature_name = [
            self.features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def recurse(node, depth, parent_rule):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                left_rule = f"{name} <= {threshold:.2f}"
                right_rule = f"{name} > {threshold:.2f}"

                if parent_rule:
                    left_rule = f"{parent_rule} AND {left_rule}"
                    right_rule = f"{parent_rule} AND {right_rule}"

                recurse(tree_.children_left[node], depth + 1, left_rule)
                recurse(tree_.children_right[node], depth + 1, right_rule)
            else:
                print(f"{indent}IF {parent_rule}:")
                print(f"{indent}  - Predict {tree_.value[node]}")

        recurse(0, 0, "")

    def print_lr_coef(self):
        coefficients = self.model.coef_

        class_labels = self.model.classes_

        for class_idx, class_label in enumerate(class_labels):
            print(f"Class '{class_label}':")
            for feature_idx, feature_name in enumerate(self.features):
                print(f"  {feature_name}: {coefficients[class_idx, feature_idx]:.4f}")


    def print_feature_correlation(self, x_label, y_label):
        # Creating a plot to visualize the distribution of 'dstport' for each class in 'type' in the full dataset
        plt.set_loglevel(level = 'warning')
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=x_label, y=y_label, data=self.data)
        plt.title(f"Correllation of {x_label} and {y_label}")
        plt.xticks(rotation=45)  # Rotate labels for better readability
        plt.show()