#%%
import random
import math
from collections import OrderedDict

import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing, model_selection, svm

random.seed()

data = pd.read_csv('contrived-loans.csv')

feature_names = data.columns.values
X = data.drop('High Risk', axis=1)
y = data['High Risk']

#%%
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.33)

parameters = {
    "kernel": ["rbf"],
    "C": [1,10,10,100,1000],
    "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }

model = sklearn.model_selection.GridSearchCV(sklearn.svm.SVC(), parameters, cv=5)
model.fit(X_train, y_train)
p = model.predict(X_test)
print('Mean Error\n')
print(sklearn.metrics.mean_squared_error(y_test, p))

#%%
def random_examples(X: pd.DataFrame, num_examples = 1) -> pd.DataFrame:
    feature_names = X.columns.values
    X_max = X.max()
    X_min = X.min()

    data = []
    for i in range(0, num_examples):
        example = OrderedDict()
        for feature_name, max_val, min_val in zip(feature_names, X_max, X_min):
            example[feature_name] = random.randint(min_val, max_val)
        data.append(example)
    df = pd.DataFrame(data)
    return df

def distance_nd(a, b):
    total = 0
    for v in range(0, len(a)):
        total += (a[v]-b[v])**2
    return math.sqrt(total)

def contrastive(model, X: pd.DataFrame, num_examples = 1):
    examples = random_examples(X, num_examples).append(X)

    p = model(X)
    predictions = p[num_examples:]
    random_predictions = p[:num_examples]

    scaler = sklearn.preprocessing.StandardScaler()
    scaled = scaler.fit_transform(examples)
    scaled_examples = scaled[num_examples:]
    random_scaled_examples = scaled[:num_examples]

    for example, prediction in zip(scaled_examples, predictions):
        min_distance = None
        closest_example = None

        for random_example, random_prediction in zip(random_scaled_examples, random_predictions):
            if prediction != random_prediction:
                distance = distance_nd(example, random_example)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    closest_example = random_example
        if closest_example is None:
            print('😡'*10)
            continue

        print(min_distance)
        print(example)
        print(closest_example)
        print(np.subtract(closest_example, example))
        print('-'*5)


contrastive(model.predict, X, 100)

#scaler = sklearn.preprocessing.StandardScaler()
#scaled_examples = scaler.fit_transform(examples.append(base))


#print(base)
#print(scaled_examples[-1,:])
#for i in range(0, len(scaled_examples)-1):
    #print(distance_nd(scaled_examples[-1,:], scaled_examples[i,:]))


#%%
