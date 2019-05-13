from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import sys
import pandas as pd

data_name = sys.argv[1]

N = int(sys.argv[2])

csv_headers = ['SMILES','Fitness']
csv = pd.read_csv(data_name,delimiter='\t',header=None, names = csv_headers)

#suppl = Chem.SmilesMolSupplier(data_name,delimiter='\t',titleLine=False,nameColumn=-1)
suppl = Chem.SmilesMolSupplier(data_name,delimiter='\t',titleLine=False,nameColumn=-1)

#charged_fragments = False
#quick = True
ms=[]

#for dat in data_list:
#	print(dat)
#	atomicNumList,charge,xyz_coordinates = x2m.read_xyz_file(dat)
#	mol = x2m.xyz2mol(atomicNumList,charge,xyz_coordinates,charged_fragments,quick)
#	#mol = Chem.RemoveHs(mol)
#	ms.append(mol)

for i in range(N):
	ms.append(suppl[i])

for m in ms:
	Chem.SanitizeMol(m) 
	tmp=AllChem.Compute2DCoords(m)

leg_list = ['{:.4f}'.format(i) for i in csv['Fitness'].tolist()]
#leg_list = csv['Fitness'].tolist()
MolDrawing.atomLabelFontSize=40
img=Draw.MolsToGridImage(ms,molsPerRow=4,subImgSize=(300,300),legends=leg_list)
img.save('images/'+data_name+'_molgrid.png')
img.show()

