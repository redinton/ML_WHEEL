
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet,Lasso,Ridge,SGDRegressor
from sklearn.linear_model import LassoCV,RidgeCV,ElasticNetCV
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR   
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

 
 class Regression_model():

 	def __init__(self,X_train,y_train,X_test,y_test):
 		self.X_train = X_train
 		self.y_train = y_train
 		self.X_test = X_test
 		self.y_test = y_test


 	def rmse_cv_train(self,model):
    	rmse= np.sqrt(-cross_val_score(model, self.X_train, self.y_train, 
    			scoring ='neg_mean_squared_error', cv = 10))
    	return(rmse)

    def rmse_cv_train(self,model):
    	rmse= np.sqrt(-cross_val_score(model, self.X_test, self.y_test, 
    			scoring ='neg_mean_squared_error', cv = 10))
    	return(rmse)

 	def regular_model(self):
		model_br = BayesianRidge()  
		model_lr = LinearRegression()  
		model_etc = ElasticNet()  
		model_las = Lasso()
		model_rid = Ridge()
		model_sgd = SGDRegressor()
		model_svr = SVR() 
		model_gbr = GradientBoostingRegressor()
		model_rfr = RandomForestRegressor()

		model_names = ['BayesianRidge','LinearRegression','ElasticNet','Lasso','Ridge',
					   'SGDRegressor','SVR','GradientBoostingRegressor','RandomForestRegressor']  

		model_dic = [model_br,model_lr,model_etc,model_las,model_rid,model_sgd,model_svr,model_gbr,model_rfr] 

		
		result_dict ={}
		for  i,clf in enumerate(model_dic):
        	value = cross_validate(clf,self.X_train,self.y_train)
        	result_dict[model_names[i]] = value
        
       	result_dict = sorted(result_dict.items(),key = lambda x:x[1],reverse = True)
		print (result_dict) 

	def ridge(self):
		
		ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
		ridge.fit(self.X_train, self.y_train)
		alpha = ridge.alpha_
		print("Best alpha :", alpha)

		print("Try again for more precision with alphas centered around " + str(alpha))
		ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
		                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
		                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
		                cv = 10)
		ridge.fit(self.X_train, self.y_train)
		alpha = ridge.alpha_
		print("Best alpha :", alpha)

		print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
		print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())

	def lasso(self):
		lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
		lasso.fit(self.X_train, self.y_train)
		alpha = lasso.alpha_
		print("Best alpha :", alpha)

		print("Try again for more precision with alphas centered around " + str(alpha))
		lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
		                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
		                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
		                          alpha * 1.4], 
		                max_iter = 50000, cv = 10)
		lasso.fit(self.X_train, self.y_train)

		alpha = lasso.alpha_
		print("Best alpha :", alpha)

		print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())
		print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())

	def elastic(self):
		elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
		elasticNet.fit(self.X_train, self.y_train)
		alpha = elasticNet.alpha_
		ratio = elasticNet.l1_ratio_
		print("Best l1_ratio :", ratio)
		print("Best alpha :", alpha )

		print("Try again for more precision with l1_ratio centered around " + str(ratio))
		elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
		                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
		                          max_iter = 50000, cv = 10)
		elasticNet.fit(self.X_train, self.y_train)
		# l1_ratio = 1 means elastic->l1_penalty
		# l1_ratio = 0 means elastic->l2_penalty

		if (elasticNet.l1_ratio_ > 1):
		    elasticNet.l1_ratio_ = 1    
		alpha = elasticNet.alpha_
		ratio = elasticNet.l1_ratio_
		print("Best l1_ratio :", ratio)
		print("Best alpha :", alpha )

		print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 
		      " and alpha centered around " + str(alpha))
		elasticNet = ElasticNetCV(l1_ratio = ratio,
		                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 
		                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 
		                                    alpha * 1.35, alpha * 1.4], 
		                          max_iter = 50000, cv = 10)

		elasticNet.fit(self.X_train, self.y_train)
		if (elasticNet.l1_ratio_ > 1):
		    elasticNet.l1_ratio_ = 1    
		alpha = elasticNet.alpha_
		ratio = elasticNet.l1_ratio_
		print("Best l1_ratio :", ratio)
		print("Best alpha :", alpha )

		print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())
		print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())


	def tune_svr(self):
		parameters = {'kernel':['rbf','linear'],
              'C':[88,89,90,91,92],
              'gamma':[0.34,0.36,0.37]}
		clf = GridSearchCV(SVR(),parameters,verbose=2)
		clf.fit(self.X_train，self.y_train)
		print (clf.best_params_)
		print (clf.best_score_)

	def tune_gbr(self):
		parameters = {'kernel':['rbf','linear'],
              'C':[88,89,90,91,92],
              'gamma':[0.34,0.36,0.37]}
		clf = GridSearchCV(GradientBoostingRegressor(),parameters,verbose=2)
		clf.fit(self.X_train，self.y_train)
		print (clf.best_params_)
		print (clf.best_score_)


	def tune_rfr(self): 
		parameters = {'kernel':['rbf','linear'],
              'C':[88,89,90,91,92],
              'gamma':[0.34,0.36,0.37]}
		clf = GridSearchCV(RandomForestRegressor(),parameters,verbose=2)
		clf.fit(self.X_train，self.y_train)
		print (clf.best_params_)
		print (clf.best_score_)



