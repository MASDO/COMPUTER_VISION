import pandas as pd
import io
import requests
import numpy as n
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

## read data
def read_data(url):
    req = requests.get(url).content
    Data = pd.read_csv(io.StringIO(req.decode('utf-8')))
    return Data
## created dataframe from

##convert data to train test samples
def convert_to_train_test(d:pd.DataFrame , train_ratio:float=0.7):
    msk = n.random.rand(len(d)) < train_ratio
    train = d[msk]
    test = d[~msk]
    return train,test
##convert data to train test samples
data_url="http://gitlab.rahnemacollege.com/rahnemacollege/tuning-registration-JusticeInWork/raw/master/dataset.csv"
data = read_data(data_url)
train_data , test_data =  convert_to_train_test(data)


def separate_x_y(data:pd.DataFrame):
    m_1,n_1 = n.shape(data)
    y = data.iloc[:, n_1 - 1]
    X = data.iloc[:, 0:n_1 - 1]
    ## print(X)
    return X , y
##X,y = (return_X_y=True)
##    return X , y
def convert_string_to_value(data:pd.DataFrame):
    m_1,n_1 = n.shape(data)
    ## print(data.dtypes)
    unique_dict = {}
    for i in range(0, n_1):
        lbl = LabelEncoder()
        if data.iloc[:, i].dtype == n.object :
           data.iloc[:, i] = lbl.fit_transform(data.iloc[:,i])
    return data


if __name__ == "__main__":
    converted_data = convert_string_to_value(data)
    converted_data = converted_data.fillna(0)
    test, train = convert_to_train_test(converted_data,0.7)
    X_tr , y_tr = separate_x_y(train)
    X_ts , y_ts = separate_x_y(test)
    print(converted_data.dtypes)
    ## print(converted_data)
    DT_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    DT_svm.fit(X_tr, y_tr)
    DT = tree.DecisionTreeClassifier()
    DT = DT.fit(X_tr, y_tr)
    score_1 = DT_svm.score(X_ts,y_ts)
    score = DT.score(X_ts, y_ts)
    print("SVMScore is: {}".format(score_1))
    print("DecisionTree Score is: {}".format(score))
