import os
import numpy as np
import pandas as pd 

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def svm(Feature,Label):

	#separate Feature and Label for train and test
	X_train, X_test, Y_train, Y_test =train_test_split(Feature, Label, test_size = 0.3)

	#make model
	Svm_Model = SVC(gamma='scale',verbose=False,decision_function_shape='ovo')
	#train model
	Svm_Model.fit(X_train, Y_train)

	#compute Accuracy
	predicted = Svm_Model.predict(X_test)
	accuracy = metrics.accuracy_score(Y_test, predicted)
	print(accuracy)

	#compute how many predict is false
	error = 0
	for i  in range(0,len(Y_test)):
		if predicted[i] != Y_test[i]:
			error+=1
	print(error)
	
	
if __name__ == '__main__':
	#read csv
	df = pd.read_csv('Feature.csv')
	#transform DataFrame to numpy.array
	DataSet=df.values 
	#print(type(DataSet))
	#make Format Feature:[[a1,b1,...,z1],.....[aN,bN,...,zN]]Label:[L1,L2,L3,...,LN]
	Feature = DataSet[:,2:]
	Label = DataSet[:,1:2]
	Label = Label.flatten()

	svm(Feature,Label)