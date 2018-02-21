

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

''' 
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(n_estimators=100,min_weight_fraction_leaf=0.2,class_weight="balanced")
	clf.fit(X_train,y_train)
'''


class metric():

	def __init__(self,y_test,y_pre_class,y_predict):
		self.y_test = y_test

	def recall(self):
		self.y_pre_class = y_pre_class
		self.y_predict = y_predict
		recall = recall_score(self.y_test, self.y_pre_class, average='binary')
		print ('Recall:\t\t'+str(recall))
		#return recall

	def accuracy(self):
		accuracy = accuracy_score(y_test, y_pre_class)
		print ('Acc:\t\t'+str(accuracy))
		#return accuracy

	def auc(self):
		auc = roc_auc_score(y_test,y_predict[:,1])
		print ('AUC:\t\t'+str(auc))
		#return auc

	def F1(self):
		f1 = f1_score(y_test, y_pre_class)
		print ('F1:\t\t'+str(f1))
		#return f1

	def all_metric(self):
		self.auc()
		self.recall()
		self.accuracy()
		self.F1()

 
	def draw_roc_curve(self):
		fpr, tpr, thresholds = roc_curve(self.y_test,self.y_predict[:,1])
	    roc_auc = auc(fpr,tpr)
	    
	    plt.title('Receiver Operating Characteristic')
	    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
	    plt.legend(loc='lower right')
	    plt.plot([0,1],[0,1],'r--')
	    plt.xlim([-0.1,1.2])
	    plt.ylim([-0.1,1.2])
	    plt.ylabel('True Positive Rate')
	    plt.xlabel('False Positive Rate')
	    plt.show()



 
def draw_feature_importance(features,clf):
    important_features = features.columns 
    feature_importance = clf.feature_importances_                                            
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)[::-1]
    # get the figure about important features
    pos = np.arange(sorted_idx.shape[0]) + .5
    #plt.subplot(1, 2, 2)
    plt.title('Feature Importance')
    plt.barh(pos, feature_importance[sorted_idx[::-1]],color='r',align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.draw()
    plt.show()





