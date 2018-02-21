import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold

class Methods():
	def __init__(self):

		pass



	def cross_validate(self,clf,X,y):
    
    	kf = KFold(n_splits=5,random_state=42)
    	corr = []
    
    	for train_index, test_index in kf.split(X,y):
        	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	        clf.fit(X_train,y_train)
	        pre = clf.predict(X_test)
	        
	        if len(pre.shape) > 1:
	            pre = pre.reshape((pre.shape[0],))
	        
	        pre = pd.Series(pre)
	        y_test = pd.Series(y_test['score']).reset_index(drop=True)

	        '''
			calculate the pearson of two Series
	        '''
	        
	        if pre.corr(y_test) is np.nan:
	            print (y_test)
	            print (pre)
	        corr.append(pre.corr(y_test))
	    #print (corr)
	    return np.mean(corr)

	'''
	fea_combine returns all combinations of the given "features"
	eg:
		all_fea = fea_combine(['cityblock','cosine'])
		all_fea: [['cityblock'],['cosine'],['cityblock','cosine']]
	'''
	def fea_combine(self,features):
	    num = len(features)
	    all_com = []
	    for i in range(1,num+1):
	        combins = [c for c in  combinations(range(num), i)]
	        #all_com.append([[ features[index] for index in a] for a in combins])
	        all_com += [[ features[index] for index in a] for a in combins]
	    return all_com

	