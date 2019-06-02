import os
import numpy as np
import pandas as pd 
import time
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,cross_val_score,ShuffleSplit,StratifiedKFold,cross_validate,GridSearchCV
from sklearn.externals import joblib

def WriteLog(log):
	now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
	LogName = "creat_SVM_model-"+now
	with open(LogName,'w') as f:
		for i in range(0,len(log)):
			f.write(log[i]+'\n')

			
#use all data ,then separate data to train and test			
def svm(Feature,Label):
	
	print("start to creat svm model")
	#separate Feature and Label for train and test
	X_train, X_test, y_train, y_test = train_test_split(Feature, Label, test_size=0.2, random_state=0)
	parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3, 1e-4,1e-5],'C': [0.1,1, 10, 100, 1000]},
				  {'kernel': ['linear'], 'C': [0.1,1, 10, 100, 1000]}]

	#make model
	Svm_Model = SVC(gamma='scale',verbose=False,decision_function_shape='ovo').fit(X_train, Y_train)
	
	#evaluate model
	accuracy = Svm_Model.score(X_test, Y_test)	   
	print("Accuracy:",accuracy)
	
	#count how many predicts are false
	predicted = Svm_Model.predict(X_test)
	error = 0
	for i  in range(0,len(Y_test)):
		if predicted[i] != Y_test[i]:
			error+=1
	print("Error:",error)
	
	log=[]
	log.append("Accuracy:"+str(accuracy))
	log.append("Error:"+str(error))
	WriteLog(log)
	
#use train data to creat model ,and save model to evaluate test data
def svm_cv(Feature,Label,Alldata):

	log=[]
	if Alldata == True:
		print("split dataset to 0.8 train and 0.2 test")
		X_train, X_test, y_train, y_test = train_test_split(Feature, Label, test_size=0.2, random_state=0)
	
	
	#some difference cross-validation methods
	skf = StratifiedKFold(n_splits=5,random_state=None, shuffle=True)
	cv = ShuffleSplit(n_splits=5, test_size=0.3)
	k_fold = KFold(n_splits=5,shuffle = True)
	
	print("start to creat svm model\n")
	parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3, 1e-4,1e-5],'C': [0.1,1, 10, 100, 1000]},
				  {'kernel': ['linear'], 'C': [0.1,1, 10, 100, 1000]}]

	clf = GridSearchCV(SVC(), parameters, scoring=None, cv=skf, verbose=0, return_train_score=False)
	if Alldata == True:
		clf.fit(X_train,y_train)
	else:
		clf.fit(Feature,Label)

	print("Best parameters set found on development set:\n")
	log.append("Best parameters set found on development set:\n")
	print(clf.best_params_)
	log.append(str(clf.best_params_)+'\n')
	print()
	
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
		log.append("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
		
	#save model
	model_name = "svm_model"
	
	if Alldata == True:	
		print("Detailed classification report:\n")
		
		y_true, y_pred = y_test, clf.predict(X_test)
		
		accuracy = clf.score(X_test, y_test)
		print("Accuracy:",accuracy)
		log.append("Accuracy:"+str(accuracy))
		
		print(classification_report(y_true, y_pred))
		log.append(classification_report(y_true, y_pred))
		
		joblib.dump(clf, model_name)
		
	else:
		BestModel = clf.best_estimator_
		#print(BestModel)
		joblib.dump(BestModel, model_name)
		
	print("The model will to be save to:",model_name)
	log.append("Name of model:"+model_name)
	WriteLog(log)
	


if __name__ == '__main__':
	print("read csv...")
	df = pd.read_csv('Train_Feature.csv')
	#transform DataFrame to numpy.array
	DataSet=df.values 
	#print(type(DataSet))
	#make Format Feature:[[a1,b1,...,z1],.....[aN,bN,...,zN]]Label:[L1,L2,L3,...,LN]
	Feature = DataSet[:,2:]
	Label = DataSet[:,1:2]
	Label = Label.flatten()
	
	Alldata = False
	svm_cv(Feature,Label,Alldata)
	#svm(Feature,Label)
	
	
	
	
#reference:
#https://martychen920.blogspot.com/2017/09/ml-gridsearchcv.html
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
#https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html