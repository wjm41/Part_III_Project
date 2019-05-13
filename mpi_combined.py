#! /usr/bin/env python
import numpy as np
from mpi4py import MPI
import soap
import os
import json
import glob
import h5py
import copy
import sys
import time
log = soap.log

data_name = sys.argv[1]
data_arg = sys.argv[2]

data_file = 'data/'+data_name+'.xyz'

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

if mpi_rank==0:
    print("Evaluating "+data_name+" on " + str(mpi_size) + " MPI processes.")

configs = soap.tools.io.read(data_file)
dat_size = len(configs)
    
def return_borders(index, dat_len):
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[index]
    border_high = mpi_borders[index+1]
    return border_low, border_high

my_border_low, my_border_high = return_borders(mpi_rank, dat_size)

data = configs[my_border_low: my_border_high]
my_len = len(data)

def evaluate_soap():
    options = soap.soapy.configure_default()
    #soap.encoder.add('Na')
    soap.encoder.add('B') 
    soap.encoder.add('Si')   
    soap.encoder.add('Se')

    dset = soap.DMapMatrixSet()
    for cidx, config in enumerate(data):
        spectrum = soap.soapy.PowerSpectrum(
                config=config, 
                options=options,
                label="config-%d" % cidx)
        dmap = spectrum.exportDMapMatrix()
        dset.append(dmap)
    output_file = 'data/mpi_'+str(mpi_rank)+'_'+data_name+'.soap'
    dset.save(output_file) 
    return dset

def evaluate_kernel(my_dset, options):

    kernel = soap.Kernel(options)
    symmetric = False
    K = np.zeros((dat_size, dat_size))
    #K = np.empty((my_len, dat_size))
    my_border_low, my_border_high = return_borders(mpi_rank, dat_size)
    my_K = np.empty((my_len,my_len))
    for index in range(mpi_size):
        start, end = return_borders(index, dat_size)
	#print(index)
        dset = soap.DMapMatrixSet("data/mpi_"+str(index)+'_'+data_name+".soap")
        K[my_border_low: my_border_high,start:end] = kernel.evaluate(my_dset, dset, symmetric, "float64")
        if index==mpi_rank:
           my_K = np.array(K[my_border_low: my_border_high,start:end])
    return np.array(K), 1/np.sqrt(my_K.diagonal())
 
# Compute SOAP descriptors
t0 = time.time()
if mpi_rank==0:
    print('Calculating SOAP descriptors...')

my_dset = evaluate_soap()

if mpi_rank==0:
    t1 = time.time()
    print "SOAP: %1.2fs" % (t1-t0)

# Compute molecular kernel
kernel_options = soap.Options()
kernel_options.set("basekernel_type", "dot")
kernel_options.set("base_exponent", 3.)
kernel_options.set("base_filter", False)
kernel_options.set("topkernel_type", data_arg)
kernel_options.set("rematch_gamma", 0.5)
kernel_options.set("rematch_eps", 1e-4)
kernel_options.set("rematch_omega", 1.0)    

if mpi_rank==0:
    t2 = time.time()
    print('Evaluating Kernel...')
    
mpi_comm.Barrier()
my_K, my_z = evaluate_kernel(my_dset,\
                    kernel_options)
print('waiting...')
mpi_comm.Barrier()

K_full = np.zeros((dat_size, dat_size))

mpi_comm.Reduce(my_K,K_full,root=0,op=MPI.SUM)

z_full = mpi_comm.gather(my_z, root=0)

if mpi_rank==0:
    z = z_full[0]
    for i in range(1,mpi_size):
        z = np.hstack((z, z_full[i]))
    t3=time.time()
    print "Unormalised Kernel: %1.2fs" % (t3-t2)
    print('Calculating outer product')
    K_full = K_full*np.outer(z,z)
    
    #real_K = np.load('data/'+data_name+'_'+data_arg+'.npy')
    #print(np.equal(K, real_K))
    np.save('data/'+data_name+'_'+data_arg+'.npy', K_full)

    t4 = time.time()
    print "Kernel: %1.2fs" % (t4-t3)
mpi_comm.Barrier()
#MPI.Finalize()
