import xplain
from model import Model

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import sklearn

np.set_printoptions(suppress=True)

data = pd.read_csv('./data/contrived-loans.csv')

feature_names = data.columns.values
X = data.drop('High Risk', axis=1)
y = data['High Risk']

model = Model(X, y, feature_names)

app = Flask(__name__)

api_name = 'xplain'

@app.route(f'/{api_name}/contrastive/variable-features', methods=['POST'])
def contrastive_variable_features_api():
    content = request.json
    X_input = pd.DataFrame(content['X'], columns=X.columns.values)
    results = xplain.contrastive(model.predict, model.X_train[:content['n_background_data']], X_input, n_samples=content['n_samples'])
    return pd.DataFrame(results).to_json(orient='records')

@app.route(f'/{api_name}/contrastive/static-features', methods=['POST'])
def contrastive_static_features_api():
    content = request.json
    X_input = pd.DataFrame(content['X'], columns=X.columns.values)
    results = xplain.contrastive_static_features(model.predict, model.X_train[:content['n_background_data']], X_input, content['static_features'], n_samples=content['n_samples'])
    return pd.DataFrame(results).to_json(orient='records')

@app.route(f'/{api_name}/contrastive/variable-features/standard-deviation', methods=['POST'])
def contrastive_standard_deviation_api():
    content = request.json
    sd = xplain.contrastive_sd(model.predict, model.X_train[:content['n_background_data']], model.X_test[content['X_index']:content['X_index']+1], n_samples=content['n_samples'], n_runs=content['n_runs'])
    return jsonify(list(sd))

@app.route(f'/{api_name}/contrastive/static-features/standard-deviation', methods=['POST'])
def contrastive_static_features_standard_deviation_api():
    content = request.json
    sd = xplain.contrastive_static_features_sd(model.predict, model.X_train[:content['n_background_data']], model.X_test[content['X_index']:content['X_index']+1], static_features=content['static_features'], n_samples=content['n_samples'], n_runs=content['n_runs'])
    return jsonify(list(sd))
