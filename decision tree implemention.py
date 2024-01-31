import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data_set = pd.read_csv('winequality-red.csv')
data_set.head(10)
data_set.hist()
data_set.info()
correlation_matrix = data_set.corr()
plt.figure(figsize=(10, 8))
heatmap = plt.pcolor(correlation_matrix, cmap='coolwarm')
plt.colorbar(heatmap)
plt.xticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns, rotation=90)
plt.yticks(np.arange(0.5, len(correlation_matrix.index), 1), correlation_matrix.index)
plt.title("Correlation Matrix")
plt.show()
sns.pairplot(data_set)
print(f'quality nums : {data_set["quality"].unique()} \nmax : {max(data_set["quality"])} \nmin : {min(data_set["quality"])} \nmean : {np.mean(data_set["quality"])} ')
quality_data = [value for value in data_set['quality']]
quality_counts = {}
for value in quality_data:
    if value in quality_counts:
        quality_counts[value] += 1
    else:
        quality_counts[value] = 1
print (quality_counts)
data_set.describe()
sns.boxplot(x = 'quality' , y = 'fixed acidity' , data = data_set)
sns.boxplot(x = 'quality', y = 'volatile acidity', data = data_set)
sns.boxplot(x = 'quality' , y= 'volatile acidity' , data = data_set)
sns.boxplot(x = 'quality' , y='citric acid' , data=data_set)
sns.boxplot(x='quality', y='residual sugar', data = data_set)
sns.boxplot(x='quality', y='chlorides', data = data_set)
sns.boxplot(x='quality', y='free sulfur dioxide', data = data_set)
sns.boxplot(x='quality', y='total sulfur dioxide', data = data_set)
sns.boxplot(x= 'quality', y = 'density', data = data_set)
sns.boxplot(x = 'quality', y = 'pH', data = data_set)
sns.boxplot(x = 'quality', y = 'sulphates', data = data_set)
data_set['Reviews'] = data_set['quality'].apply(lambda x: 1.0 if 1 <= x <= 3 else (2.0 if 4 <= x <= 7 else (3.0 if 8 <= x <= 10 else None)))
class CustomStandardScaler:
    def __init__(self, columns=None):
        self.columns = columns
        self.mean_values = None
        self.std_values = None

    def fit(self, data):
        if self.columns is None:
            self.columns = data.columns

        self.mean_values = data[self.columns].mean()
        self.std_values = data[self.columns].std()

    def transform(self, data):
        scaled_data = (data[self.columns] - self.mean_values) / self.std_values
        return pd.concat([data.drop(columns=self.columns), scaled_data], axis=1)

def main():
    x_columns_to_scale = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol' , 'quality']
    x = data_set[x_columns_to_scale]
    y = data_set['Reviews']

    x_scaler = CustomStandardScaler(columns=x_columns_to_scale)

    x_scaler.fit(x)

    x_scaled = x_scaler.transform(x)
    return x_scaled, y

if __name__ == "__main__":
    x, y = main()

class CustomTrainTestSplit:
    def __init__(self, test_size=0.25, random_state=None):
        self.test_size = test_size
        self.random_state = random_state
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def split(self, X, y):
        data = np.column_stack((x, y))
        if self.random_state is not None:
            np.random.seed(self.random_state)
        np.random.shuffle(data)

        split_index = int((1 - self.test_size) * len(data))

        self.x_train, self.x_test = data[:split_index, :-1], data[split_index:, :-1]
        self.y_train, self.y_test = data[:split_index, -1], data[split_index:, -1]

        return self.x_train, self.x_test, self.y_train, self.y_test

custom_splitter = CustomTrainTestSplit(test_size=0.25, random_state=42)
x_train, x_test, y_train, y_test = custom_splitter.split(x, y)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

model = DecisionTreeClassifier()
model.fit(x_train , y_train)
model_predict = model.predict(x_test)
dt_acc_score = accuracy_score(y_test, model_predict)
print(dt_acc_score*100)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
class Node:
    def __init__(self, feature=None, value=None, result=None):
        self.feature = feature  
        self.value = value      
        self.result = result   
        self.children = {}      

class DecisionTreeID3:
    def __init__(self):
        self.root = None

    def fit(self, X, y, features=None):
        if features is None:
            features = list(range(X.shape[1]))
        self.root = self._build_tree(X, y, features)

    def _build_tree(self, X, y, features):
        unique_classes, counts = np.unique(y, return_counts=True)

        if len(unique_classes) == 1:
            return Node(result=unique_classes[0])

        if len(features) == 0:
            majority_class = unique_classes[np.argmax(counts)]
            return Node(result=majority_class)

        best_feature = self._choose_best_feature(X, y, features)
        node = Node(feature=best_feature)

        for value in np.unique(X[:, best_feature]):
            subset_indices = X[:, best_feature] == value
            subset_X = X[subset_indices]
            subset_y = y[subset_indices]

            if subset_X.shape[0] == 0:
                majority_class = unique_classes[np.argmax(counts)]
                node.children[value] = Node(result=majority_class)
            else:
                remaining_features = [f for f in features if f != best_feature]
                node.children[value] = self._build_tree(subset_X, subset_y, remaining_features)

        return node

    def _choose_best_feature(self, X, y, features):
        info_gains = {feature: self._calculate_information_gain(X[:, feature], y) for feature in features}
        return max(info_gains, key=info_gains.get)

    def _calculate_information_gain(self, feature, target):
        entropy_before_split = self._calculate_entropy(target)
        unique_values = np.unique(feature)

        weighted_entropy_after_split = 0
        for value in unique_values:
            subset_indices = feature == value
            weighted_entropy_after_split += np.sum(subset_indices) / len(feature) * self._calculate_entropy(target[subset_indices])

        information_gain = entropy_before_split - weighted_entropy_after_split
        return information_gain

    def _calculate_entropy(self, target):
        class_probabilities = np.bincount(target) / len(target)
        entropy = -np.sum(p * np.log2(p) for p in class_probabilities if p > 0)
        return entropy

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.root
            while node.children:
                feature_value = sample[node.feature]
                if feature_value not in node.children:
                    break
                node = node.children[feature_value]
            predictions.append(node.result)
        return predictions
model = DecisionTreeID3()
model.fit(x_train, y_train_encoded)
model_predict1 = model.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
dt_acc_score = accuracy_score(y_test, model_predict1)
print(dt_acc_score*100)

