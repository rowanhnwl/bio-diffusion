from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

sdf_path = "grid_searches/no_constraint/caco2 permeability_ts-1000_iw-0_fw-0_ai-5_am-add_sm-none/caco2 permeability_ts-1000_iw-0_fw-0_ai-5_am-add_sm-none/01272025_18_29_52_mol.sdf"

df = LoadSDF(sdf_path, smilesName='SMILES')

try:
    smiles = df["SMILES"]
except:
    print("No available SMILES strings")
    smiles = []

print(len(smiles))
print(len(set(smiles)))