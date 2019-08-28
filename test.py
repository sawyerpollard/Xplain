#%%
import xplain
from model import Model

import pandas as pd
import numpy as np
import sklearn

np.set_printoptions(suppress=True)

#%%
data = pd.read_csv('./data/contrived-loans.csv')

feature_names = data.columns.values
X = data.drop('High Risk', axis=1)
y = data['High Risk']

#%%
model = Model(X, y, feature_names)
print(model.predict(model.X_test))

#%%
results = xplain.contrastive(model.predict, model.X_train[:500], model.X_test[:100], n_samples=1000)

#%%

for result in results:
    print(result['min_distance'])
    print(result['prediction'])

    print(result['feature_names'])
    print(result['example'])
    print(result['closest_example'])
    print(result['contrast_example'])

    print(result['inverse_example'])
    print(result['inverse_closest_example'])
    print(result['inverse_contrast_example'])
    print('-'*10)

for i in range(4, 5):
    n_background_data = 500
    X_index = 102
    n_samples = 10**i
    n_runs = 5
    sd = xplain.contrastive_sd(model.predict, model.X_train[:n_background_data], model.X_test[X_index:X_index+1], n_samples, n_runs)

    print('n_samples='+str(n_samples))
    print('n_runs='+str(n_runs))
    print('n_background_data='+str(n_background_data))
    print('X_index='+str(X_index))
    print(feature_names)
    print(sd)
    print('-'*10)