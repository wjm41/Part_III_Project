import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import gpflow as gpf
import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error 
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.figsize': [6.4, 5.4]})

data_name = sys.argv[1]
data_arg = sys.argv[2]
K = np.load(data_name+'_'+data_arg+'.npy')

csv_name = data_name+'.csv'

if data_name=='lipo':
	csv_headers=['Name','lip_val','SMILES']
	pred_val = 'lip_val'
elif data_name=='esol':	
	csv_headers=['Name','pred_ESOL','min_deg','mol_weight','num_H_donor','num_rings','num_rot_bonds','surface_area','expt_ESOL','SMILES']
	pred_val = 'expt_ESOL'
elif data_name=='FreeSolv':
	csv_headers=['Name','SMILES','expt','calc']
	pred_val = 'expt'
elif data_name=='CatS':
	csv_headers=['SMILES','val','Name']
	pred_val = 'val'
else:
   print('Unrecognised data file')

len_dataset = len(K[0])

#Split into features - dataset specific
X=np.array(range(len_dataset))
X=X.astype(int)
X=X.reshape(-1,1)
Y=np.zeros(len_dataset)
Y=Y.reshape(-1,1)
kernel_mat = K+1
csv = pd.read_csv(csv_name, header=None,names=csv_headers)

j=0
for i,row in csv.iterrows():
	#print(row['Name'])
	C_count=row['SMILES'].count('C')+row['SMILES'].count('c')-row['SMILES'].count('Cl')
	#N_count=row['SMILES'].count('N')+row['SMILES'].count('n')
	#O_count=row['SMILES'].count('O')+row['SMILES'].count('o')
	#Y[j] = [C_count, N_count, O_count]
	Y[j] = C_count
	j+=1

#Data_set split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10)

kernel_diag = tf.constant(np.diag(kernel_mat))
kernel_mat = tf.constant(kernel_mat)
def soap_sub(A,A2):
	global kernel_mat
	A = tf.cast(A,tf.int32)
	A2 = tf.cast(A2,tf.int32)
	K_rows = tf.gather_nd(kernel_mat, A)
	K_rows = tf.transpose(K_rows)
	K_mat = tf.gather_nd(K_rows, A2)
	K_mat = tf.transpose(K_mat)
	return tf.cast(K_mat, tf.float64)


class SOAP(gpf.kernels.Kernel):
	global kernel_mat, kernel_diag
	def __init__(self):
		super().__init__(input_dim=1, active_dims=[0])
		self.variance = gpf.Param(1.0, transform=gpf.transforms.Exp())
		self.power = gpf.Param(1.0, transform=gpf.transforms.Exp())
		#self.mag = gpf.Param(1.0, transform=gpf.transforms.Exp())
	
	@gpf.params_as_tensors
	def K(self, A, A2=None):
		if A2 is None:
			A2=A
		K_mat = soap_sub(A,A2)
		#return self.mag*tf.math.exp(-2*(1-K_mat)*self.variance)
		return tf.math.pow(self.variance*K_mat, self.power)
		#return self.variance*K_mat
	def Kdiag(self, A):
		A=tf.cast(A,tf.int32)
		K_diag = tf.cast(tf.gather_nd(kernel_diag, A),tf.float64)
		#return self.variance*K_diag
		return tf.math.pow(self.variance*K_diag, self.power)
		#return self.mag*tf.math.exp(-2*(1-K_diag)*self.variance)

k_soap = SOAP()
k_noise = gpf.kernels.White(0.1)
k=k_soap+k_noise
#k=k_soap
#k = mk.SharedIndependentMok(k, output_dimensionality = 3)

# initialisation of inducing input locations (M random points from the training inputs)
#Z = X_train.copy() 
#feature = mf.SharedIndependentMof(gpf.features.InducingPoints(Z))

#model = gpf.models.SVGP(X_train,y_train, k, gpf.likelihoods.Gaussian(), feat = feature)
#model = gpf.models.SVGP(X_train,y_train, k, gpf.likelihoods.Gaussian(),Z) 

model = gpf.models.GPR(X_train,y_train, k) 
opt =gpf.train.ScipyOptimizer()
opt.minimize(model)

#print(type(X_train[0][0]))
#print(type(X_test[0][0]))

#mean and variance GP prediction
y_pred, y_var = model.predict_y(X_test)
y_pred = np.round(y_pred)
len_pred = len(y_pred)

for i in range(len_pred):
	if y_pred[i]<0:
		y_pred[i]=0.0

#for j in range(3):
#	for i in range(len_pred):
#		if y_pred[i,j]<0:
#			y_pred[i,j]=0.0

#Output score
#score = np.zeros(3)
#for i in range(len(score)):
#	score[i] = mean_absolute_error(y_test[:,i],y_pred[:,i])
 
score = mean_absolute_error(y_test,y_pred)
print("C MAE: {:.2f}%".format(score)) 
#print("N MAE: {:.2f}%".format(score[1]))
#print("O MAE: {:.2f}%".format(score[2]))
#atoms = ['C', 'N', 'O']

def reg_plot(y_test, y_pred):
	#Plot y_pred against y_true
	plt.clf()
	plt.scatter(y_test, y_pred, s=10, marker='o',c=np.sqrt(y_var), cmap='bwr')
	plt.plot(y_test,y_test, linestyle='--', color='k')
	cbar = plt.colorbar()
	cbar.set_label('GP st_dev')
	plt.xlabel("y_true")
	plt.ylabel("y_pred")
	#plt.title('predicted vs actual '+atoms[arg]+' atom values\nusing GP regression on SOAP average kernel')
	plt.savefig(data_name+'_reg_SVGP_C.pdf')
	return
def mae_plot(y_true, y_predicted, y_var, atoms, arg):
	y_test = y_true[:,arg]
	y_pred = y_predicted[:,arg]
	y_var = y_var[:,arg]
	plt.clf()
	MAE=np.zeros(len(y_test)-1)
	r2 = np.zeros(len(y_test)-1)
	#var_val=np.zeros(len(y_test)-1)
	x_max = np.sqrt(np.amax(y_var))
	x_min = np.sqrt(np.amin(y_var))
	var_val=np.linspace(100,0,len(y_test)-1)
	for i in range(len(y_test)-1):
		max_ind = y_var.argmax()
		y_test=np.delete(y_test,max_ind)
		y_pred=np.delete(y_pred,max_ind,0)
		MAE[i]=mean_absolute_error(y_test,y_pred)
		r2[i]=accuracy_score(y_test,y_pred)*100
		y_var=np.delete(y_var,max_ind)
	print('Max Accuracy = {:.2f}'.format(np.amax(r2)))
	plt.plot(var_val, MAE, linestyle='--',marker='o',color='k',markersize=4)
	#plt.xlabel('GP st_dev')
	plt.xlabel('% of data used, sorted by uncertainty')
	plt.ylabel('MAE')
	#plt.xlim([x_min,x_max])
	plt.xlim([0,100])
	plt.title(atoms[arg]+' MAE against st_dev threshold\n'+data_name+' using SOAP '+data_arg+' kernel')
	plt.gca().invert_xaxis()
	plt.savefig(data_name+'_mae_SVGP_'+atoms[arg]+'.pdf')
	
	plt.clf()
	plt.plot(var_val, r2,linestyle='--',marker='o',color='k',markersize=4)
	plt.xlabel('GP st_dev')
	plt.ylabel('Accuracy %')
	plt.ylim([0,100])
	#plt.xlim([x_min,x_max])
	plt.xlim([0,100])
	plt.gca().invert_xaxis()
	plt.title(atoms[arg]+'Accuracy against st_dev threshold'+'\n'+data_name+' using SOAP '+data_arg+' kernel')
	plt.savefig(data_name+'_acc_SVGP_'+atoms[arg]+'.pdf')

reg_plot(y_test,y_pred)
#for i in range(3):
#	print('Plotting for '+atoms[i])
#	reg_plot(y_test, y_pred)
	#mae_plot(y_test, y_pred, y_var, atoms, i)
