import pandas as pd
import sys

csv_name = 'HIV.csv'
smiles_name='HIV.can'

SMILES_df = pd.read_csv(csv_name, header=None,names=['SMILES','Val'],index_col=False)
file=open(smiles_name,'w')
for i,row in SMILES_df.iterrows():
	file.write(row['SMILES']+'\t'+'#'+str(i)+'\n')
file.close()
