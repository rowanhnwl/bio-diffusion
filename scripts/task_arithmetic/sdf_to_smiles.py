from rdkit.Chem.PandasTools import LoadSDF
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn", type=str, required=True)
    args = parser.parse_args()

    df = LoadSDF(args.fn, smilesName='SMILES')

    print(df["SMILES"])