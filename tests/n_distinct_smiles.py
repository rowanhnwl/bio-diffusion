from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

sdf_path = "output/geom_grid_search_caco2/caco2 permeability_GEOM_ts-1000_iw-0.75_fw-0.01_ai-3_am-mean_sm-exp_caco2 permeability_GEOM_ts-1000_iw-0.75_fw-0.01_ai-3_am-mean_sm-exp/caco2 permeability_GEOM_ts-1000_iw-0.75_fw-0.01_ai-3_am-mean_sm-exp_0/02142025_12_47_44_mol.sdf"

df = LoadSDF(sdf_path, smilesName='SMILES')

print(df)

try:
    smiles = df["SMILES"]
except:
    print("No available SMILES strings")
    smiles = []

print(len(smiles))
print(len(set(smiles)))