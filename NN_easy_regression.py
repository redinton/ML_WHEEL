

import tensorflow as tf
from sklearn.model_selection import KFold
import itertools

class NN_regression():

	def __init__(self,X_train,y_train,X_test,y_test):
 		self.X_train = X_train
 		self.y_train = y_train
 		self.X_test = X_test
 		self.y_test = y_test

	def input_fn(X,y, pred = False):
	    if pred == False:
	        feature_cols = {k: tf.constant(X[k].values) for k in FEATURE_COLUMNS}
	        labels = tf.constant(y.values) 
	        return feature_cols, labels

	    if pred == True:
	        feature_cols = {k: tf.constant(X[k].values) for k in FEATURE_COLUMNS}
	        return feature_cols

	def prepare(self,FEATURE_COLUMNS):
		feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_COLUMNS]
		config = tf.contrib.learn.RunConfig(gpu_memory_fraction=0.3, log_device_placement=True)
		regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
		                                          dropout = 0.1,
		                                          #optimizer = tf.train.AdamOptimizer(),
		                                          hidden_units=[64,128],
		                                          model_dir='./models/dnnregressor')

		clf = regressor
		clf.fit(input_fn=lambda: self.input_fn(self.X_train,self.y_train), steps=2000)
		predict = clf.predict(input_fn=lambda: self.input_fn(self.X_test,self.y_test,True))
		predictions = list(itertools.islice(predict, X_test.shape[0]))
	    pre = pd.Series(predictions)


	FEATURE_COLUMNS = ['cityblocktf_idf', 'cosinetf_idf', 'euclideantf_idf', 'l1tf_idf',
	       'l2tf_idf', 'manhattantf_idf', 'wm_distance', 'w2vc_euclidean',
	       'w2vc_cosine', 'w2vc_l2', 'simple_cosine', 'simple_euclidean',
	       'simple_l2']

	

	def cross_validate(self,clf,X,y):
    
    	kf = KFold(n_splits=5,random_state=42)
    	corr = []
    
		for train_index, test_index in kf.split(X,y):
	        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	        
	        clf = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
	                                          #dropout = 0.2,
	                                          #optimizer = tf.train.AdamOptimizer(),
	                                          hidden_units=[64,128],
	                                          model_dir='./models/dnnregressor')
	        
	        clf.fit(input_fn=lambda: self.input_fn(X_train,y_train), steps=200)
	        
	        ev = clf.evaluate(input_fn=lambda: self.input_fn(X_test,y_test), steps=1)
	        #print('ev: {}'.format(ev))
	        print ('*'*80)
	        predict = clf.predict(input_fn=lambda: self.input_fn(X_test,y_test,True))
	        predictions = list(itertools.islice(predict, X_test.shape[0]))
	        
	        pre = pd.Series(predictions)
	        y_test = pd.Series(y_test['score']).reset_index(drop=True)
        
        	if pre.corr(y_test) is np.nan:
            	print (y_test)
            	print (pre)
        	print (pre.corr(y_test))
        	corr.append(pre.corr(y_test))
    	print (corr)
    
    	return np.mean(corr),clf
    

NN_regression = NN_regression()
