import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing, model_selection, svm

class Model:
    def __init__(self, X, y, feature_names):
        self.feature_names = feature_names
        
        self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(X, y, test_size=.33)

        parameters = {
            "kernel": ["rbf"],
            "C": [1,10,10,100,1000],
            "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            }

        self.model = sklearn.model_selection.GridSearchCV(sklearn.svm.SVC(), parameters, cv=5)
        self.model.fit(self.X_train, self.y_train)
        p = self.model.predict(self.X_test)
        print('Mean Error\n')
        print(sklearn.metrics.mean_squared_error(self.y_test, p))

    def has_membership(self, data, result):
        rule = 'Has membership'

        if data['Membership']:
            result['trace'].append(rule)
            return self.loan_cutoff(data, result)
        else:
            result['trace'].append('!' + rule)
            result['out'] = None
            return result

    def too_young(self, data, result):
        rule = 'Age at least 18'

        if data['Age'] >= 18:
            result['trace'].append(rule)
            return self.has_membership(data, result)
        else:
            result['trace'].append('!' + rule)
            result['out'] = None
            return result

    def loan_cutoff(self, data, result):
        rule = 'Loan amount less than 7000'

        if data['Loan Amount'] < 7000:
            result['trace'].append(rule)
            return self.loan_model(data, result)
        else:
            result['trace'].append('!' + rule)
            result['out'] = None
            return result

    def loan_model(self, data, result):
        rule = 'Passed model'

        if (self.model.predict(data.to_numpy().reshape(1, -1))[0] == 0):
            result['trace'].append(rule)
            result['out'] = 0
            return result
        else:
            result['trace'].append('!' + rule)
            result['out'] = None
            return result

    def root(self, data):
        result = self.too_young(data, { 'trace': [] })
        return result

    def predict_explain(self, X):
        data = pd.DataFrame(X, columns = self.feature_names[:-1])
        out = []

        for i in range(0, len(data)):
            result = self.root(data.iloc[i])
            if result['out'] is None:
                result['out'] = 1
            out.append(result)        

        return out

    def predict(self, X):
        out = []

        for result in self.predict_explain(X):
            if result['out'] is None:
                result['out'] = 0
            out.append(result['out'])
            
        return np.array(out)
        