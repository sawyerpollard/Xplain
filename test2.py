#%%
import numpy as np
import pandas as pd
import sklearn
import shap

shap.initjs()

data = pd.read_csv('contrived-loans.csv')

feature_names = data.columns.values
X = data.drop('High Risk', axis=1)
y = data['High Risk']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.33)

#%%
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
def has_membership(data, result):
    rule = 'Has membership'

    if data['Membership']:
        result['trace'].append(rule)
        return loan_cutoff(data, result)
    else:
        result['trace'].append('!' + rule)
        result['out'] = None
        return result

def too_young(data, result):
    rule = 'Age at least 18'

    if data['Age'] >= 18:
        result['trace'].append(rule)
        return has_membership(data, result)
    else:
        result['trace'].append('!' + rule)
        result['out'] = None
        return result

def loan_cutoff(data, result):
    rule = 'Loan amount less than 7000'

    if data['Loan Amount'] < 7000:
        result['trace'].append(rule)
        return loan_model(data, result)
    else:
        result['trace'].append('!' + rule)
        result['out'] = None
        return result

def loan_model(data, result):
    rule = 'Passed model'

    rule_explanation = shap_key_features(explainer, data, feature_names=feature_names)

    if (model.predict(data.to_numpy().reshape(1, -1))[0] == 0):
        result['trace'].append(rule + '\n' + rule_explanation)
        result['out'] = 0
        return result
    else:
        result['trace'].append('!' + rule + '\n' + rule_explanation)
        result['out'] = None
        return result

def root(data):
    result = too_young(data, { 'trace': [] })
    return result

#%%
explainer = shap.KernelExplainer(model.predict, X_train)

def shap_key_features(explainer, X, feature_names, max_features = 1):
    shap_values = explainer.shap_values(X)
    absolute_sum = np.sum(np.absolute(shap_values))

    max = np.amax(shap_values)
    min = np.amin(shap_values)

    max_index = np.where(shap_values == max)[0][0]
    min_index = np.where(shap_values == min)[0][0]

    max_importance = max/absolute_sum * 100
    min_importance = np.absolute(min)/absolute_sum * 100

    max_string = f'{feature_names[max_index]} played a {round(max_importance, 1)}% role in not passing.'
    min_string = f'{feature_names[min_index]} played a {round(min_importance, 1)}% role in passing.'

    return max_string + '\n' + min_string

#%%
def umodel(X):
    data = pd.DataFrame(X, columns = feature_names[:-1])
    out = []

    for i in range(0, len(data)):
        result = root(data.iloc[i])
        if result['out'] is None:
            result['out'] = 1
        out.append(result)        

    return out

def print_umodel(umodel_out):
    for x in umodel_out:
        for y in x['trace']:
            print(y + '\n')
        print('REJECT' if x['out'] else 'ACCEPT')
        print('\n')
        print('-'*100)

print_umodel(umodel(X_train[:20]))


#%%
def shap_umodel(X):
    data = pd.DataFrame(X, columns = feature_names[:-1])
    out = []

    for i in range(0, len(data)):
        result = root(data.iloc[i])
        if result['out'] is None:
            result['out'] = 0

        out.append(result['out'])
    
    return np.array(out)
shap_umodel(X_train)


#%%
def shap1():
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test[0])

    print(shap_values)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names[:-1], max_display=50)
    shap.force_plot(explainer.expected_value, shap_values, X_test, feature_names=feature_names[:-1])
shap1()
def shap2():
    explainer = shap.KernelExplainer(umodel, X_train[:100])
    shap_values = explainer.shap_values(X_test[:100], nsamples=50)

    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names[:-1], max_display=50)
    shap.force_plot(explainer.expected_value, shap_values, X_test[:100], feature_names=feature_names[:-1])

#%%
shap2()

#%%
shap.summary_plot(shap_values, X, plot_type="bar")
