import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import gpflow
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 

data_name = sys.argv[1]
data_arg = sys.argv[2]

kernel_mat = np.load(data_name+'_'+data_arg+'.npy')

#print(np.amin(kernel_mat))
reg_name = data_name+'_'+data_arg+'_reg.pdf'
err_name = data_name+'_'+data_arg+'_rmse.pdf'
r2_name = data_name+'_'+data_arg+'_r2.pdf'
csv_name = data_name+'.csv'

if data_name=='lipo_final':
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

len_dataset = len(kernel_mat[0])

kernel_mat+=1 #improves performance (?)

#Split into features - dataset specific
X=np.array(range(len_dataset))
X=X.astype(int)
X=X.reshape(-1,1)

Y=np.zeros(len_dataset)
Y=Y.reshape(-1,1)

kernel_mat=np.float64(kernel_mat)
csv = pd.read_csv(csv_name, header=None,names=csv_headers)

j=0
for i,row in csv.iterrows():
	Y[j]=row[pred_val]
	j+=1

kernel_diag = tf.constant(np.diag(kernel_mat),dtype=tf.float64)
kernel_mat = tf.constant(kernel_mat,dtype=tf.float64)

def soap_sub(A,A2):
	global kernel_mat
	A = tf.cast(A,tf.int32)
	A2 = tf.cast(A2,tf.int32)
	K_rows = tf.gather_nd(kernel_mat, A)
	K_rows = tf.transpose(K_rows)
	K_mat = tf.gather_nd(K_rows, A2)
	K_mat = tf.transpose(K_mat)
	return tf.cast(K_mat, tf.float64)


class SOAP(gpflow.kernels.Kernel):
	global kernel_mat, kernel_diag
	def __init__(self):
		super().__init__(input_dim=1, active_dims=[0])
		self.variance = gpflow.Param(1.0, transform=gpflow.transforms.Exp())
		#self.power = gpflow.Param(1.0, transform=gpflow.transforms.Exp())
		self.mag = gpflow.Param(1.0, transform=gpflow.transforms.Exp())
	
	@gpflow.params_as_tensors
	def K(self, A, A2=None):
		if A2 is None:
			A2=A
		K_mat = soap_sub(A,A2)
		return self.mag*tf.math.exp(-2*(1-K_mat)*self.variance)
		#return tf.math.pow(self.variance*K_mat, self.power)
		#return self.variance*K_mat
	def Kdiag(self, A):
		A=tf.cast(A,tf.int32)
		K_diag = tf.cast(tf.gather_nd(kernel_diag, A),tf.float64)
		#return self.variance*K_diag
		#return tf.math.pow(self.variance*K_diag, self.power)
		return self.mag*tf.math.exp(-2*(1-K_diag)/self.variance)

k_soap = SOAP()
k_noise = gpflow.kernels.White(0.1)
k = k_soap+k_noise

#custom_config = gpflow.settings.get_settings()
#custom_config.numerics.jitter_level = 1e-3
#with gpflow.settings.temp_settings(custom_config), gpflow.session_manager.get_session().as_default():

#Data_set split
from sklearn.model_selection import train_test_split
r2_list = []
rmse_list = []
for i in range(1,2):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=i)

	model = gpflow.models.GPR(X_train,y_train,kern=k) 
	opt = gpflow.train.ScipyOptimizer()
	opt.minimize(model)

	#mean and variance GP prediction
	y_pred, y_var = model.predict_y(X_test)

	#Output score
	score = r2_score(y_test, y_pred)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
 
	print("R^2: {:.3f}".format(score))
	print("RMSE: {:.3f}".format(rmse))

	r2_list.append(score)
	rmse_list.append(rmse)

	#Plot y_pred against y_true
	plt.rcParams.update({'font.size': 16})
	plt.rcParams.update({'figure.figsize': [6.4, 5.4]})
	np.savetxt(data_name+'_'+data_arg+'_ytest.dat',y_test)
	np.savetxt(data_name+'_'+data_arg+'_ypred.dat',y_pred)
	np.savetxt(data_name+'_'+data_arg+'_yvar.dat',np.sqrt(y_var))
	#plt.scatter(y_test, y_pred, s=10, marker='o',c=np.sqrt(y_var), cmap='bwr')
	#plt.plot(y_test, y_test, linestyle='--', color='k')
	#cbar = plt.colorbar()
	#cbar.set_label('GP st_dev')
	#plt.xlabel("y_true")
	#plt.ylabel("y_pred")
	#plt.savefig(reg_name)
r2_list = np.array(r2_list)
rmse_list = np.array(rmse_list)
print("mean R^2: {:.4f}".format(np.mean(r2_list)))
print("R^2 standard error: {:.4f}".format(np.std(r2_list)/np.sqrt(len(r2_list))))
print("mean RMSE: {:.4f}".format(np.mean(rmse_list)))
print("RMSE standard error: {:.4f}".format(np.std(rmse_list)/np.sqrt(len(rmse_list))))

#plt.clf()
#MAE=np.zeros(len(y_test)-1)
#r2 = np.zeros(len(y_test)-1)
#var_val=np.linspace(100,0,len(y_test)-1)
#x_max = np.sqrt(np.amax(y_var))
#x_min = np.sqrt(np.amin(y_var))
#for i in range(len(y_test)-1):
#	max_ind = y_var.argmax()
#	y_test=np.delete(y_test,max_ind)
#	y_pred=np.delete(y_pred,max_ind,0)
#	MAE[i]=np.sqrt(mean_squared_error(y_test,y_pred))
#	r2[i]=r2_score(y_test,y_pred)
#	y_var=np.delete(y_var,max_ind)

#print('Max R2 = {:.3f}'.format(np.amax(r2)))
#plt.plot(var_val, MAE, linestyle='--',marker='o',color='k',markersize=4)
#plt.xlabel('% data threshold by st_dev')
#plt.ylabel('RMSE')
#plt.title('RMSE against st_dev threshold\n'+data_name+' using SOAP '+data_arg+' kernel')
#plt.gca().invert_xaxis()
#plt.savefig(err_name)

#plt.clf()
#plt.plot(var_val, r2,linestyle='--',marker='o',color='k',markersize=4)
#plt.xlabel('% data thresold by st_dev')
#plt.ylabel(r'$R^2$')
#plt.ylim([0,1])
#plt.gca().invert_xaxis()
#plt.title(r'$R^2$ against st_dev threshold'+'\n'+data_name+' using SOAP '+data_arg+' kernel')
#plt.savefig(r2_name)
