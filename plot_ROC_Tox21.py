import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
from scipy import interp
data_arg = sys.argv[1]

data_name = ['nr_ar_noion','nr_er_noion','nr_aromatase_noion','nr_ppar_gamma_noion','nr_er_lbd_noion','nr_ar_lbd_noion','nr_ahr_noion']
pdf_name ='Tox21_NR_'+data_arg+'_ROC.pdf'

#data_name = ['MUV_466','MUV_548','MUV_600','MUV_712','MUV_713']
#pdf_name = 'MUV_'+data_arg+'ROC.pdf'


mat=np.empty([1,1])

def soap_kernel(X, Y):
	global mat
	"""
	Calleable function that returns soap kernel
	"""
	return np.dot(np.dot(X,mat), Y.T)

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve

fpr_list = []
tpr_list = []
t1=time.time()

tprs = []
base_fpr = np.linspace(0, 1, 101)

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.figsize': [6.4, 5.4]})

for i in data_name:
	kernel_mat = np.load(i+'_'+data_arg+'.npy')
	can_name = i+'.can'

	len_dataset = len(kernel_mat[0])

	#Split into features - dataset specific
	X=np.zeros((len_dataset, len_dataset))
	Y=np.zeros(len_dataset)
	can = pd.read_csv(can_name, delimiter='\t',header=None,names=['SMILES','name','label'])

	for i,row in can.iterrows():
		X[i][i]=1
		Y[i]=int(row['label'])
		
	#Data_set split, train SVM
	X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2)
	
	mat=kernel_mat
	#Call SVM
	svc = SVC(C=1,kernel = soap_kernel).fit(X_train, y_train)

	fpr,tpr,thresholds = roc_curve(y_test, svc.decision_function(X_test))
	plt.plot(fpr, tpr, color='grey',lw=1.5)
	tpr = interp(base_fpr, fpr, tpr)
	tpr[0] = 0.0
	tprs.append(tpr)
t2=time.time()
print('Time = {:.1f}s'.format(t2-t1))
tprs = np.array(tprs)
tpr_mean = tprs.mean(axis=0)
plt.plot(base_fpr ,tpr_mean, color='r',lw=1.5)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig(pdf_name)
plt.show()
