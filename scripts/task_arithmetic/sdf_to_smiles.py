from rdkit import Chem
from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem import Descriptors, Crippen
from matplotlib import pyplot as plt

import os

if __name__ == "__main__":
    outpath = "output"

    logps = []

    for dirname in os.listdir(outpath):

        if "lipophilicity" not in dirname:
            continue

        dirpath = os.path.join(outpath, dirname)

        filename = os.listdir(dirpath)[0]
        filepath = os.path.join(dirpath, filename)

        df = LoadSDF(filepath, smilesName='SMILES')

        for i, smi in enumerate(df["SMILES"]):

            mol = Chem.MolFromSmiles(smi)
            logp = Crippen.MolLogP(mol)

            logps.append(logp)

for i, l in enumerate(logps):
    plt.scatter(i, l)

plt.show()