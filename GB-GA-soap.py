from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from itertools import compress

import numpy as np
import pandas as pd
import random
import time
import sys

import soap
import sys
#from mpi4py import MPI

import pybel
import crossover as co
import mutate as mu

p_table = {6:'C', 7:'N', 8:'O',9:'F',16:'S'}

#mpi_comm = MPI.COMM_WORLD
#mpi_rank = mpi_comm.Get_rank()
#mpi_size = mpi_comm.Get_size()

soap.soapy.PowerSpectrum.verbose = False
soap.soapy.wrap.PowerSpectrum.verbose = False
soap.encoder.add('B') 
soap.encoder.add('Si')   
soap.encoder.add('Se')

# Compute molecular kernel
kernel_options = soap.Options()
kernel_options.set("basekernel_type", "dot")
kernel_options.set("base_exponent", 3.)
kernel_options.set("base_filter", False)
#kernel_options.set("topkernel_type", "average")
kernel_options.set("topkernel_type", "rematch")
kernel_options.set('rematch_gamma',0.5)
kernel_options.set('rematch_eps',1e-4)
kernel_options.set('rematch_omega',1.0)

input_mol = sys.argv[1]
dataset = sys.argv[2]
max_gen = int(sys.argv[3])
n = int(sys.argv[4])

mol = pybel.readfile('xyz','../data/'+input_mol+'.xyz').next()

co.average_size = mol.exactmass
co.size_stdev = 5.0

atom_types = []
test_mol = []
for atom in mol:
	atom_types.append(p_table[atom.atomicnum])
	test_mol.append(list(atom.coords))

num_atom = len(atom_types)
C = Chem.MolFromSmarts('[#6]')
N = Chem.MolFromSmarts('[#7]') 
O = Chem.MolFromSmarts('[#8]')
F = Chem.MolFromSmarts('[#9]')
S = Chem.MolFromSmarts('[#16]')

atom_nums = [0,0,0,0,0]
for atom in atom_types:
    if atom=='C': atom_nums[0]+=1
    elif atom=='N': atom_nums[1]+=1
    elif atom=='O': atom_nums[2]+=1
    elif atom=='F': atom_nums[3]+=1
    else: atom_nums[4]+=1
num_C = atom_nums[0]
num_N = atom_nums[1]
num_O = atom_nums[2]
num_F = atom_nums[3]
num_S = atom_nums[4]

def read_file(file_name):
  mol_list = []
  name_list = []
  csv_headers=[]  
  if dataset=='esol':
     csv_headers=['Name','pred_ESOL','min_deg','mol_weight','num_H_donor','num_rings','num_rot_bonds','surface_area','expt_ESOL','SMILES']
  elif dataset=='lipo':
       csv_headers=['Name','lip_val','SMILES']
  else: 
       print('help')
  csv = pd.read_csv(file_name, header=None,names=csv_headers) 
  for i,row in csv.iterrows():
      mol_list.append(Chem.MolFromSmiles(row['SMILES']))
      name_list.append(row['Name'])
  return np.array(mol_list), np.array(name_list)

def write_file(population,file_name,tag=0.000,original=True):
  global atom_types
  global test_mol
  f = open(file_name,'w')
  if original==True:
      len_mol = len(atom_types)
      f.write(str(len_mol)+'\n')
      f.write('smiles="original" tag="original"\n')
      for i in range(len_mol):
          f.write(atom_types[i][0]+'\t'+str(test_mol[i][0])+'\t'+str(test_mol[i][1])+'\t'+str(test_mol[i][2])+'\n')

  for m in population:
      types=[]
      len_mol = 0
      for atom in m.GetAtoms():
          types.append(atom.GetSymbol())
          len_mol+=1
      m = Chem.AddHs(m)
      conf = AllChem.EmbedMolecule(m, maxAttempts=1000, useRandomCoords = True)
      m = Chem.RemoveHs(m)
      coords = np.zeros((len_mol,3))
      if conf==0:
         coords=m.GetConformer().GetPositions() 
      f.write(str(len_mol)+'\n')
      f.write('smiles="'+Chem.MolToSmiles(m)+'" tag="'+str(tag)+'"\n')
      for i in range(len_mol):
          f.write(types[i][0]+'\t'+str(coords[i][0])+'\t'+str(coords[i][1])+'\t'+str(coords[i][2])+'\n')

def evaluate_soap(configs, options, output_file=None, save=False):
  dset = soap.DMapMatrixSet()
  for cidx, config in enumerate(configs):
      spectrum = soap.soapy.PowerSpectrum(
          config=config,
          options=options,
          label="config-%d" % cidx)
      dmap = spectrum.exportDMapMatrix()
      dset.append(dmap)
  if save==True:
     dset.save(output_file)
  return dset

def evaluate_kernel(x_set, y_set):
  global kernel_options
  kernel = soap.Kernel(kernel_options)
  symetric = False
  K = kernel.evaluate(x_set, y_set, symetric, "float64")
  z = 1./np.sqrt(K.diagonal())
  K = K*np.outer(z,z)
  return K

def pop_fitness(population, file_name, gen):
  global test_mol
  global input_mol
  
  write_file(population, file_name, original=True)

  configs = soap.tools.io.read(file_name)

  # Compute SOAP descriptors
  options = soap.soapy.configure_default()
  ref_set = soap.DMapMatrixSet("../data/"+input_mol+".soap")
  dset = evaluate_soap(configs, options)
 
  K = evaluate_kernel(dset, dset)

  fitness = K[0][1:]
 
  mol_len = np.empty_like(fitness)
  C_len = np.empty_like(fitness)
  N_len = np.empty_like(fitness)
  O_len = np.empty_like(fitness)
  F_len = np.empty_like(fitness)
  S_len = np.empty_like(fitness)
  for ind, mol in enumerate(population):
      mol_len[ind] = abs(mol.GetNumAtoms() - num_atom)
      C_len[ind] = abs(len(mol.GetSubstructMatches(C)) - num_C)
      N_len[ind] = abs(len(mol.GetSubstructMatches(N)) - num_N)
      O_len[ind] = abs(len(mol.GetSubstructMatches(O)) - num_O)
      F_len[ind] = abs(len(mol.GetSubstructMatches(F)) - num_F)
      S_len[ind] = abs(len(mol.GetSubstructMatches(S)) - num_S)

  fitness = fitness - gen*0.0001*(C_len+N_len+O_len+F_len+S_len)

  global max_score

  if np.amax(fitness) > max_score[0]:
    max_score=[np.amax(fitness), Chem.MolToSmiles(population[np.argmax(fitness)])]
  
  global fit_mean, fit_std
  fitness = np.maximum((fitness - fit_mean)/fit_std, 0.0)
  
  fitness = fitness/np.sum(fitness)
  
  return fitness

def make_initial_population(population_size,file_name):
  mol_list, name_list = read_file(file_name)
  
  #esol_set = soap.DMapMatrixSet("../data/esol_final.soap") 

  #K = evaluate_kernel(ref_set, esol_set)
  
  fitness = np.load('../data/'+dataset+'_'+input_mol+'_rematch.npy')
  
  inds = fitness.argsort()

  sorted_fitness = np.flip(fitness[inds])
  sorted_mol = np.flip(mol_list[inds])
  sorted_names = np.flip(name_list[inds])
 
  #ASSUMES THE CORRECT MOLECULE IS IN THE LIST - from 2nd as 1st typically too close 
  population = sorted_mol[1:population_size+1]
  pop_names = sorted_names[1:population_size+1]
  pop_fit = sorted_fitness[1:population_size+1]

  pop_mean = np.mean(pop_fit)
  pop_std = np.std(pop_fit)

  f = open('init_pop_'+dataset+'_'+input_mol+'.dat','w')
  print('Initial population:')
  for i in range(population_size):
      print(pop_names[i],pop_fit[i],population[i].GetNumAtoms())
      f.write(Chem.MolToSmiles(population[i])+'\t'+str(pop_fit[i])+'\n')
  
  return population, pop_mean, pop_std

def make_mating_pool(population,fitness):
  mating_pool = []
  for i in range(population_size):
  	mating_pool.append(np.random.choice(population, p=fitness))

  return mating_pool
 

def reproduce(mating_pool,population_size,mutation_rate):
  new_population = []
  for n in range(population_size):
    parent_A = random.choice(mating_pool)
    parent_B = random.choice(mating_pool)
    #print Chem.MolToSmiles(parent_A),Chem.MolToSmiles(parent_B)
    new_child = co.crossover(parent_A,parent_B)
    #print new_child
    if new_child != None:
	    new_child = mu.mutate(new_child,mutation_rate)
	    #print("after mutation",new_child)
	    if new_child != None:
	    	new_population.append(new_child)

  
  return new_population


global max_score

#SA_scores = np.loadtxt('SA_scores.txt')
#SA_mean =  np.mean(SA_scores)
# SA_std=np.std(SA_scores)

population_size = n 
generations = max_gen
mutation_rate = 0.2

print('population_size', population_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('')

file_name = '../data/'+dataset+'_'+input_mol+'.csv'

global fit_mean, fit_std

t0 = time.time()
population, fit_mean, fit_std = make_initial_population(population_size,file_name)


print('\nNo of atoms in '+input_mol+' = '+str(num_atom))

max_score = [-99999.,'']

f = open('champion_'+dataset+'_'+input_mol+'.dat','w')
for generation in range(generations):
  print('\nGeneration #'+str(generation+1))
  
  fitness = pop_fitness(population, '../data/GB-GA-SOAP-'+dataset+'_'+input_mol+'.xyz', generation)
  mating_pool = make_mating_pool(population,fitness)
  population = reproduce(mating_pool,population_size,mutation_rate)
  
  print('fitness = '+str(max_score[0]), max_score[1], 'num_atoms = '+str(Chem.MolFromSmiles(max_score[1]).GetNumAtoms()))
  f.write(max_score[1]+'\t'+ str(max_score[0])+'\n')
  f.flush()
t1 = time.time()
print('')
print('time ',t1-t0)
