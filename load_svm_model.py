import os
import numpy as np
import pandas as pd 
import time
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import classification_report
def evaluate(Feature,Label):

    clf = joblib.load('svm_model')
    print("Detailed classification report:\n")
	
    y_true, y_pred = Label, clf.predict(Feature)
    print(classification_report(y_true, y_pred))
	
    
if __name__ == '__main__':

    df = pd.read_csv('Test_Feature.csv')
    #transform DataFrame to numpy.array
    DataSet=df.values 
    
    #make Format Feature:[[a1,b1,...,z1],.....[aN,bN,...,zN]]Label:[L1,L2,L3,...,LN]
    Feature = DataSet[:,2:]
    Label = DataSet[:,1:2]
    Label = Label.flatten()
    
    evaluate(Feature,Label)

#reference:
#https://martychen920.blogspot.com/2017/09/ml-gridsearchcv.html