import random
import math
from collections import OrderedDict

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True)

def gen_random_examples(X: pd.DataFrame, n_examples = 1000) -> pd.DataFrame:
    feature_names = X.columns.values

    data = []
    for i in range(0, n_examples):
        example = OrderedDict()
        for feature_name, max_val, min_val in zip(feature_names, X.max(), X.min()):
            example[feature_name] = random.randint(min_val, max_val)
        data.append(example)
    df = pd.DataFrame(data)
    return df

def distance_nd(a, b):
    total = 0
    for v in range(0, len(a)):
        total += (a[v]-b[v])**2
    return math.sqrt(total)

def contrastive(model, data: pd.DataFrame, X: pd.DataFrame, n_samples = 1000):
    feature_names = X.columns.values

    examples = gen_random_examples(data, n_samples).append(X)

    p = model(examples)
    predictions = p[n_samples:]
    random_predictions = p[:n_samples]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(examples)
    scaled_examples = scaled[n_samples:]
    random_scaled_examples = scaled[:n_samples]

    results = []
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
            #print('😡'*10)
            min_distance = 0
            closest_example = example

        contrast_example = np.subtract(closest_example, example)

        abs_max = np.max(np.absolute(contrast_example))
        abs_max_index = np.where(np.absolute(contrast_example) == abs_max)[0][0]

        inverse_example = scaler.inverse_transform(example)
        inverse_closest_example = scaler.inverse_transform(closest_example)
        inverse_contrast_example = np.subtract(inverse_closest_example, inverse_example)

        result = {}
        result['contrast_example'] = contrast_example
        result['closest_example'] = closest_example
        result['abs_max'] = abs_max
        result['abs_max_index'] = abs_max_index
        result['inverse_example'] = inverse_example
        result['inverse_closest_example'] = inverse_closest_example
        result['inverse_contrast_example'] = inverse_contrast_example
        result['min_distance'] = min_distance
        result['prediction'] = prediction
        result['feature_names'] = feature_names
        result['example'] = example

        results.append(result)

        #print(min_distance)
        #print(prediction)

        #print(feature_names)
        #print(example)
        #print(closest_example)
        #print(contrast_example)

        #print(inverse_example)
        #print(inverse_closest_example)
        #print(inverse_contrast_example)
        #print('-'*10)
    return results

def contrastive_static_features(model, data: pd.DataFrame, X: pd.DataFrame, static_features = [], n_samples = 1000):
    def gen_static_examples(examples: pd.DataFrame, static_example: pd.DataFrame, static_features):
        examples = examples.copy()
        for static_feature in static_features:
            examples[static_feature] = static_example[static_feature]
        return examples
    
    feature_names = X.columns.values

    scaler = sklearn.preprocessing.StandardScaler()

    random_examples = gen_random_examples(data, n_samples)
    
    predictions = model(X)
    
    results = []
    for (_, example), prediction in zip(X.iterrows(), predictions):
        min_distance = None
        closest_example = None

        static_random_examples = gen_static_examples(random_examples, example, static_features)

        random_predictions = model(static_random_examples)

        scaled = scaler.fit_transform(static_random_examples.append(example))
        example = scaled[-1:][0]
        static_random_examples = scaled[:-1]

        for random_example, random_prediction in zip(static_random_examples, random_predictions):
            if prediction != random_prediction:
                distance = distance_nd(example, random_example)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    closest_example = random_example
        if closest_example is None:
            #print('😡'*10)
            min_distance = 0
            closest_example = example

        contrast_example = np.subtract(closest_example, example)

        abs_max = np.max(np.absolute(contrast_example))
        abs_max_index = np.where(np.absolute(contrast_example) == abs_max)[0][0]

        inverse_example = scaler.inverse_transform(example)
        inverse_closest_example = scaler.inverse_transform(closest_example)
        inverse_contrast_example = np.subtract(inverse_closest_example, inverse_example)

        result = {}
        result['contrast_example'] = contrast_example
        result['closest_example'] = closest_example
        result['abs_max'] = abs_max
        result['abs_max_index'] = abs_max_index
        result['inverse_example'] = inverse_example
        result['inverse_closest_example'] = inverse_closest_example
        result['inverse_contrast_example'] = inverse_contrast_example
        result['min_distance'] = min_distance
        result['prediction'] = prediction
        result['feature_names'] = feature_names
        result['example'] = example

        results.append(result)

        #print(min_distance)
        #print(prediction)

        #print(feature_names)
        #print(example)
        #print(closest_example)
        #print(contrast_example)

        #print(inverse_example)
        #print(inverse_closest_example)
        #print(inverse_contrast_example)
        #print('-'*10)
    return results

def contrastive_sd(model, data, x, n_samples = 1000, n_runs = 30):
    out = []
    
    #print('n_samples='+str(n_samples))
    #print('n_runs='+str(n_runs))
    #print('n_background_data='+str(n_background_data))
    #print('X_index='+str(X_index))

    for i in range(0, n_runs):
        c = contrastive(model, data, x, n_samples)
        out.append(c[0]['inverse_contrast_example'])
    sd = np.std(out, axis=0)

    return sd

def contrastive_static_features_sd(model, data, x, static_features = [], n_samples = 1000, n_runs = 30):
    out = []
    
    #print('n_samples='+str(n_samples))
    #print('n_runs='+str(n_runs))
    #print('n_background_data='+str(n_background_data))
    #print('X_index='+str(X_index))

    for i in range(0, n_runs):
        c = contrastive_static_features(model, data, x, static_features, n_samples)
        out.append(c[0]['inverse_contrast_example'])
    sd = np.std(out, axis=0)

    return sd