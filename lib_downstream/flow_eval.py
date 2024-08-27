
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config_dpsyn
from parameter_parser import parameter_parser
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import _tree
from model_debugger import ModelDebugger


# Renaming the class to FlowClassifier
class FlowClassifier:
    def __init__(self, data_path, model):
        self.logger = logging.getLogger('FlowClassifier')
        self.data_path = data_path
        self.model = model
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.logger.info("loading data from " + self.data_path + " and model is " + self.model.__class__.__name__)

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.dropna()

    def preprocess_data(self):
        self.features = ["srcport", "dstport", "proto", "td", "pkt", "byt"]
        self.target = "type"

        # Encoding the 'proto' categorical field
        label_encoder = LabelEncoder()
        self.data['proto'] = label_encoder.fit_transform(self.data['proto'])

        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.X_train = train_data[self.features]
        self.y_train = train_data[self.target]
        self.X_test = test_data[self.features]
        self.y_test = test_data[self.target]

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self):
        self.logger.info("train model")
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        self.logger.info("evaluate model")
        predictions = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)    

# Test driver function
def main(args):
    os.chdir("../../")
    
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    file_prefix = args['dataset_name']

    datasets = {
        file_prefix + 'Raw': config_dpsyn.RAW_DATA_PATH + file_prefix + '.csv',
        file_prefix + 'Syn': config_dpsyn.SYNTHESIZED_RECORDS_PATH + ('_'.join((args['dataset_name'], str(args['epsilon']))) + '.csv'),
    }

    models = [
        DecisionTreeClassifier(random_state=42),
        LogisticRegression(max_iter=1000, random_state=42),
        RandomForestClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),
        MLPClassifier(max_iter=1000, random_state=42)
    ]

    results = []

    for dataset_name, data_path in datasets.items():
        for model in models:
            classifier = FlowClassifier(data_path, model)
            classifier.load_data()
            classifier.preprocess_data()
            classifier.train_model()
            accuracy = classifier.evaluate_model()
            results.append({
                'Model': model.__class__.__name__,
                'Dataset': dataset_name,
                'Accuracy': accuracy
            })
            print(f'Model: {model.__class__.__name__}, Dataset: {dataset_name}, Accuracy: {accuracy}')
            debugger = ModelDebugger(classifier.model, classifier.data, classifier.features, classifier.target)

            if model.__class__.__name__ == 'DecisionTreeClassifier':
                pass
            elif model.__class__.__name__ == 'LogisticRegression':
                debugger.print_lr_coef()
                debugger.print_feature_correlation('type', 'dstport')

    
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == "__main__":
    args = parameter_parser()
    main(args)

