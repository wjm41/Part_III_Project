import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score

data_name = sys.argv[1]
data_arg = sys.argv[2]

kernel_mat = np.load(data_name+'_'+data_arg+'.npy')
pdf_name = data_name+'_ROC.pdf'
can_name = data_name+'.can'

def soap_kernel(X, Y):
    """
    Calleable function that returns soap kernel
    """
    return np.dot(np.dot(X, kernel_mat), Y.T)

len_dataset = len(kernel_mat[0])

kernel_mat=np.float64(kernel_mat)
X=np.zeros((len_dataset, len_dataset))
Y=np.zeros(len_dataset)

can = 0
if data_name[:2]=='MU':
     csv_headers=['SMILES','Name','label']
     can = pd.read_csv(can_name, header=None,delimiter='\t', names=csv_headers)
elif data_name[:2]=='nr':
     csv_headers = ['SMILES','Name','label']
     can = pd.read_csv(can_name, header=None, delimiter='\t',names=csv_headers)
elif data_name[:2]=='HI':
     csv_headers = ['SMILES','label']
     can = pd.read_csv(can_name, header=None,names=csv_headers)
else:
     csv_headers = 0
pred_val = 'label'


print('Loading data...')

for i,row in can.iterrows():
	X[i][i]=1
	Y[i]=int(row[pred_val])


print('Beginning test runs...')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,average_precision_score

ROC_list = []
#PRC_list = []
for i in range(5):	
	#Data_set split, train SVM
	X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=i)

	print('Training SVM, run #{}'.format(str(i+1)))

	#Call SVM
	svc = SVC(C=1,kernel = soap_kernel ).fit(X_train, y_train)
	ROC_score =  roc_auc_score(y_test,svc.decision_function(X_test))
	ROC_list.append(ROC_score)
	#PRC_score = average_precision_score(y_test,svc.decision_function(X_test))
	#PRC_list.append(PRC_score)
	print("AUC: {:.3f}".format(ROC_score))
	#print("AUC-PRC: {:.3f}".format(PRC_score))

ROC_list = np.array(ROC_list)
#PRC_list = np.array(PRC_list)


print('Mean AUC-ROC: {:.3f}'.format(np.mean(ROC_list)))
#print('Mean AUC-PRC: {:.3f}'.format(np.mean(PRC_list)))
#cross_scores = cross_val_score(svc, X, Y, cv=5)
#print(cross_scores)

#Plot ROC
#from sklearn.metrics import roc_curve

#fpr,tpr,thresholds = roc_curve(y_test, svc.decision_function(X_test))
#plt.plot(fpr, tpr, label="ROC Curve")
#plt.xlabel("FPR")
#plt.ylabel("TPR")
#plt.title('ROC Curve for '+data_name+' dataset')
#plt.savefig(pdf_name)
