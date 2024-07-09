import os
import sys
import argparse
import pandas as pd
from sm import SmallMoleculeSimilarity2D
from sm import SmallMoleculeSimilarity3D
from sm import GenerateFigures

import warnings

import os
import csv
import glob
import argparse
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.Chem import rdchem
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import Draw
from rdkit.Chem import DataStructs
from rdkit.Chem.AllChem import AddHs
import numpy as np
import time


# Filter out the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)



def myparser():
    parser = argparse.ArgumentParser(description='Scripts to run X-SMSim')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input SMILES CSV file')
    parser.add_argument('--rd2path', '-rd2path', type=str, required=True, help='Path to 2D Reference Database File (SMILES file)')
    parser.add_argument('--rd3_PDBpath', '-rd3_PDBpath', type=str, required=True, help='Path to 3D Reference Database containing PDBs')
    parser.add_argument('--rd3_SDFpath', '-rd3_SDFpath', type=str, required=True, help='Path to 3D Reference Database containing SDFs')
    parser.add_argument('--type', '-type', choices=['pdb', 'sdf'], default='pdb',  type=str, required=True, help='Ref DB file format type for 3D structures')
    parser.add_argument('--outdir', '-o', type=str, required=True, help='Output Directory')
    return parser

def main(arglist):
    parser = myparser()
    args = parser.parse_args(arglist)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)


    smobject = SmallMoleculeSimilarity2D(sm_input_file=args.input, ref_path=args.rd2path, outdir=args.outdir)
    smobject.read_csv()
    smobject.indata
    smobject.read_reflist()
    smobject.ref_smiles[1:5]
    smobject.smiles_to_fingerpring_with_check_for_invalid()
    
    # write morgan figer print to the outdir
    # fp_output_file_path = os.path.join(smobject.outdir, f"fingerprints_2D_{smobject.filebase}.csv")
    fp_output_file_path = os.path.join(smobject.outdir, "fingerprints_2D_" + smobject.filebase + ".csv")
    d2_morgan_fp = pd.DataFrame.from_dict(smobject.d2_valid_input_mols, orient='index')
    d2_morgan_fp.index.name = 'SMILES'
    d2_morgan_fp.reset_index(inplace=True)
    d2_morgan_fp.to_csv(fp_output_file_path, index=False)


    smobject.apply_tanimoto()
    # write matix to the outdir
    #tanimoto_outfile = os.path.join(smobject.outdir, f"tanimoto_sim_Matrix_2D_{smobject.filebase}.csv")
    tanimoto_outfile = os.path.join(smobject.outdir, "tanimoto_sim_Matrix_2D_" + smobject.filebase + ".csv")
    smobject.d2_tanimoto_cofficient_matrix.to_csv(tanimoto_outfile, index=True)  # Save as CSV

    # Writing output to the outdir
    # Write the invalid molecules to a text file
    if len(smobject.d2_invalid_mols) > 0:
        with open(os.path.join(smobject.outdir, "output_smiles_invalid_molecule_2D_" + smobject.filebase + ".txt"), "w") as f:
            f.write("\n".join(smobject.d2_invalid_mols))


    sm3d= SmallMoleculeSimilarity3D(inputdata=smobject.indata, outdir=smobject.d3_outdir, filebase=smobject.filebase, pdb_moad_refpath= args.rd3_PDBpath, pdb_sdf_refpath=args.rd3_SDFpath)
    sm3d.indata
    sm3d.d3_outdir
    sm3d.generate_3D_coodinates()
    sm3d.d3_mol_dict
    sm3d.d3_invalid_molecules
    df_obs = sm3d.get_fingerprint_pdb(sm3d.d3_mol_dict)

    if args.type == "pdb":
        ref_chem_dict_pdb = sm3d.get_mol_pdb(directory=sm3d.PDB_MOAD, extension="pdb", type="pdb")
        df_ref_pdb = sm3d.get_fingerprint_pdb(ref_chem_dict_pdb)
        sm3d.apply_tanimoto_coefficient_for_df(df_ref_pdb, df_obs)
        output_file = sm3d.d3_outdir + "/" + sm3d.filebase + "_3DSim_pdb.csv"
        sm3d.d3_tanimoto_sim.to_csv(output_file, index=True)  # Save as CSV
    elif args.type == "sdf":
        ref_chem_dict_sdf = sm3d.get_mol_sdf(directory=sm3d.PDB_SDF, extension="sdf", type="sdf")
        df_ref_sdf = sm3d.get_fingerprint_sdf(ref_chem_dict_sdf)
        sm3d.apply_tanimoto_coefficient_for_df(df_ref_sdf, df_obs)
        output_file = sm3d.d3_outdir + "/" + sm3d.filebase + "_3DSim_sdf.csv"
        sm3d.d3_tanimoto_sim.to_csv(output_file, index=True)  # Save as CSV
    else:
        print("Invalid type specified")

    sm3d.generate_RDKitDescriptors()
    output_file_rdk_desp = sm3d.d3_outdir + "/" + sm3d.filebase + "_RDKitDescriptors.csv"
    sm3d.RDKitDescriptors.to_csv(output_file_rdk_desp, index=True)  # Save as CSV
    sm3d.normalized_rdkit_descriptors()

    sm3d.generate_MORDREDdescriptors()
    output_file_mord_desp = sm3d.d3_outdir + "/" + sm3d.filebase + "_MORDREDDescriptors.csv"
    sm3d.MORDREDdescriptors.to_csv(output_file_mord_desp, index=True)  # Save as CSV
    sm3d.normalized_mordreddescriptors()

    # generating plots for 2d and 3d    
    fig_object = GenerateFigures(inputdata=smobject.indata, d2_morgan_fp=d2_morgan_fp, d3_morgan_fp=df_obs, normalized_RDKitDescriptors= sm3d.normalized_RDKitDescriptors , normalized_MORDREDdescriptors= sm3d.normalized_MORDREDdescriptors, outdir=sm3d.d3_outdir)
    fig_object.plot_2d()
    fig_object.plot_3d()
    fig_object.plot_normalized_RDKitDescriptors()
    fig_object.plot_tsne_hc_RDKitDescriptors()
    fig_object.plot_normalized_MORDREDdescriptors()
    fig_object.plot_tsne_hc_MORDREDdescriptors()


if __name__ == '__main__':
    main(sys.argv[1:])


