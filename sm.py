import os
import csv
import glob
from glob import glob
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
import pandas as pd

from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem import PDBWriter
from rdkit.DataStructs import TanimotoSimilarity, ConvertToNumpyArray
from rdkit import DataStructs
from rdkit.Chem import SDMolSupplier
from rdkit.Chem import Descriptors3D, rdmolfiles
from mordred import Calculator, descriptors

from sklearn.preprocessing import MinMaxScaler

#cmap = plt.cm.get_cmap('tab20', len(labels.unique()))
import matplotlib.cm as cm

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from rdkit.Chem import Descriptors3D, rdmolfiles


class SmallMoleculeSimilarity2D:
    '''
    Class to compute 2D similarity bewteen small molecules
    '''
    def __init__(self, sm_input_file, ref_path, outdir):
        '''
        Initialize input file name
        '''
        self.inputfile = sm_input_file
        self.refpath = ref_path
        self.filebase = os.path.splitext(sm_input_file)[0]
        self.outdir=outdir
        self.d3_outdir = os.path.join(outdir, f"{self.filebase}_pdb")
        self.indata = None
        self.ref_smiles = None
        self.d2_valid_input_mols = None
        self.d2_valid_ref_mols = None
        self.d2_invalid_mols = None
        self.d2_tanimoto_cofficient_matrix = None
        self.forcefield = rdForceFieldHelpers.UFFGetMoleculeForceField

    def read_csv(self):
        ''''
        Read in input file as csv
        '''
        self.indata = pd.read_csv(self.inputfile)

    def read_reflist(self):
        ''''
        Read reflist
        '''
        with open(self.refpath, "r") as f:
            ref_smiles = f.read().splitlines()
            self.ref_smiles = ref_smiles
        
    def smiles_to_fingerpring_with_check_for_invalid(self):
        '''
        Convert SMILES to RDKit molecules and check for invalid molecules. Calculate Morgan fingerprints for all valid molecules
        '''

        def calculate_morgan_fingerprint(mol):
            fp = AllChem.GetMorganFingerprintAsBitVect(AddHs(mol), 2, nBits=1024)
            fp_len = len(fp.GetOnBits())
            arr = np.zeros((1024,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            num_zeros_added = 1024 - fp_len
            fp_list = list(arr)
            return fp_list
        
        invalid_smiles = [smi for smi in self.indata['SMILES'] if MolFromSmiles(smi) is None]
        invalid_ref_smiles = [smi for smi in self.ref_smiles if MolFromSmiles(smi) is None]

        valid_ref_mols = {smi: MolFromSmiles(smi) for smi in self.ref_smiles if MolFromSmiles(smi) is not None}
        valid_input_mols = {smi: {'rdchem_Mol': MolFromSmiles(smi),'Morgan_Fingerprint': calculate_morgan_fingerprint(MolFromSmiles(smi))} for smi in self.indata['SMILES'] if MolFromSmiles(smi) is not None}
       
        self.d2_invalid_mols = invalid_smiles + invalid_ref_smiles
        self.d2_valid_ref_mols = valid_ref_mols
        self.d2_valid_input_mols = valid_input_mols
            
    def calculate_tanimoto(self, mol1, mol2):
        '''
        Function to get Morgan fingerprint and add Hs and calculate Tanimoto coefficient between two molecules
        '''
        fp1 = AllChem.GetMorganFingerprintAsBitVect(AddHs(mol1), 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(AddHs(mol2), 2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def apply_tanimoto(self):
        '''
        Apply calculate_tanimoto to list of molecules from input and reference
        '''
        matrix = np.zeros((len(self.d2_valid_input_mols.keys()), len(self.d2_valid_ref_mols.keys())))
        for i, key1 in enumerate(self.d2_valid_input_mols):
            for j, key2 in enumerate(self.d2_valid_ref_mols):
                # Retrieve the corresponding values from the dictionaries
                mol1 = self.d2_valid_input_mols[key1]['rdchem_Mol']
                mol2 = self.d2_valid_ref_mols[key2]
                # Perform the desired comparison and store the result in the matrix
                comparison_value = self.calculate_tanimoto(mol1, mol2)
                matrix[i,j] = comparison_value

        # Get the column and row names from the dictionaries
        col_names = list(self.d2_valid_ref_mols.keys())
        row_names = list(self.d2_valid_input_mols.keys())

        # Create a DataFrame from the matrix with the column and row names
        df_d2_tanimoto = pd.DataFrame(matrix, index=row_names, columns=col_names)
        self.d2_tanimoto_cofficient_matrix = df_d2_tanimoto


class SmallMoleculeSimilarity3D:
    '''
    Class to compute 3D similarity bewteen small molecules
    '''
    def __init__(self, inputdata, outdir, filebase, pdb_moad_refpath, pdb_sdf_refpath):
        '''
        Initialize input file name
        '''
        self.d3_outdir = outdir
        self.indata = inputdata
        self.d3_mol_dict = None
        self.d3_invalid_molecules = None
        self.forcefield = rdForceFieldHelpers.UFFGetMoleculeForceField
        self.filebase = filebase
        self.PDB_MOAD = pdb_moad_refpath
        self.PDB_SDF = pdb_sdf_refpath
        self.bv_len= 1024 # Initialize the bitvector length


    def generate_3D_coodinates(self):
        '''
        Generate the 3D coordinates for each molecule
        '''
        invalid_molecules = []
        output_mappings = {}

        for i, row in self.indata.iterrows():
            smiles = row["SMILES"]
            mol_name = row["UniqueMolecules"]
            # Generate the RDKit molecule from the SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_molecules.append(smiles)
                continue
            # Generate the 3D coordinates
            AllChem.EmbedMolecule(mol)
            # Optimize the 3D coordinates using the UFF force field
            try:
                self.forcefield(mol)
            except ValueError:
                invalid_molecules.append(smiles)
                continue
            output_mappings[mol_name] = mol

        self.d3_mol_dict = output_mappings
        self.d3_invalid_molecules = invalid_molecules

        # Assuming self.d3_outdir is the directory path where the files should be stored
        output_directory = self.d3_outdir

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Write each record to a separate file in the output directory
        for mol_name, mol in output_mappings.items():
            # Generate the output filename using the mol_name and output_directory
            output_filename = os.path.join(output_directory, f"{mol_name}.pdb")

            # Write the PDB structure to a file
            with open(output_filename, "w") as f:
                writer = PDBWriter(f)
                writer.write(mol)
                writer.close()

        # Write the invalid molecules to a text file
        invalid_molecules_filename = os.path.join(output_directory, f"{os.path.basename(self.filebase)}_d3_invalid_molecules.txt")
        with open(invalid_molecules_filename, "w") as f:
            f.write("\n".join(self.d3_invalid_molecules))

    def get_mol_pdb(self, directory, extension, type):
        file_dict = {}
        invalid_molecules = []
        for file_name in os.listdir(directory):
            if file_name.endswith(extension):
                file_path = os.path.join(directory, file_name)
                file_basename = os.path.splitext(file_name)[0]
                mol = Chem.MolFromPDBFile(file_path)
                if mol is None:
                    invalid_molecules.append(file_basename)
                    continue
                file_dict[file_basename] = mol
        invalid_molecule_file_name = os.path.join(self.d3_outdir, f"{type}_d3_invalid_molecules.txt")
        with open(invalid_molecule_file_name, 'w') as f:
            for mol_name in invalid_molecules:
                f.write(mol_name + '\n')
        return file_dict
    
    def get_mol_sdf(self, directory, extension, type):
        file_dict = {}
        invalid_molecules = []
        for file_name in os.listdir(directory):
            if file_name.endswith(extension):
                file_path = os.path.join(directory, file_name)
                file_basename = os.path.splitext(file_name)[0]
                # Read the SDF file using SDMolSupplier
                supplier2 = SDMolSupplier(file_path)
                # Loop through the molecules in the SDF file
                for mol in supplier2:
                    if mol is None:
                        invalid_molecules.append(file_basename)
                        continue
                file_dict[file_basename] = mol
        invalid_molecule_file_name = os.path.join(self.d3_outdir, f"{type}_d3_invalid_molecules.txt")
        with open(invalid_molecule_file_name, 'w') as f:
            for mol_name in invalid_molecules:
                f.write(mol_name + '\n')
        return file_dict
        
    
    def get_fingerprint_pdb(self, dictobject):
        df_list = []
        for item in dictobject:
            mol = dictobject[item]
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.bv_len)
            # Convert the fingerprint to a numpy array
            arr = np.zeros((self.bv_len,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fp_list = list(arr)
            # Create a DataFrame for the current molecule
            df_item = pd.DataFrame({'Name': [item], 'Fingerprint': [fp], 'FP': [fp_list]})
            df_list.append(df_item)
        # Concatenate all the DataFrames into a single DataFrame
        df = pd.concat(df_list, ignore_index=True)
        return df
    

    def get_fingerprint_sdf(self, dictobject):
        df_list = []
        for item in dictobject:
            mol = dictobject[item]
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.bv_len)
                arr = np.zeros((self.bv_len,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fp_list = list(arr)
                df_item = pd.DataFrame({'Name': [item], 'Fingerprint': [fp], 'FP': [fp_list]})
                df_list.append(df_item)
        df = pd.concat(df_list, ignore_index=True)
        return df


    def apply_tanimoto_coefficient_for_df(self, ref_df, obs_df):
        # Initialize the output matrix
        out_mat = pd.DataFrame(columns=obs_df['Name'])
        # Loop through the fingerprints in the first data frame
        for i, fp1 in enumerate(ref_df['Fingerprint']):
            # Get the molecule name
            name1 = ref_df.loc[i, 'Name']
            # Initialize the row for this molecule in the output matrix
            row = pd.DataFrame(index=[name1], columns=obs_df['Name'])
            # Loop through the fingerprints in the second data frame
            for j, fp2 in enumerate(obs_df['Fingerprint']):
                # Get the molecule name
                name2 = obs_df.loc[j, 'Name']
                # Calculate the Tanimoto coefficient
                sim = TanimotoSimilarity(fp1, fp2)
                # Add the similarity to the output matrix
                row.loc[name1, name2] = sim
            # Add the row to the output matrix
            out_mat = out_mat.append(row)
        self.d3_tanimoto_sim = out_mat

    
    def generate_RDKitDescriptors(self):
        rdk_dict = dict(PDB_File=[], Asphericity=[], Eccentricity=[], Inertial_Shape_Factor=[], NPR1=[], NPR2=[], PMI1=[], PMI2=[], PMI3=[], Radius_of_Gyration=[], Spherocity_Index=[])
        for item in self.d3_mol_dict:
            mol = self.d3_mol_dict[item]
            rdk_dict['PDB_File'].append(item)
            rdk_dict['Asphericity'].append( Descriptors3D.Asphericity(mol))
            rdk_dict['Eccentricity'].append(Descriptors3D.Eccentricity(mol))
            rdk_dict['Inertial_Shape_Factor'].append(Descriptors3D.InertialShapeFactor(mol))
            rdk_dict['NPR1'].append(Descriptors3D.NPR1(mol))
            rdk_dict['NPR2'].append(Descriptors3D.NPR2(mol))
            rdk_dict['PMI1'].append(Descriptors3D.PMI1(mol))
            rdk_dict['PMI2'].append(Descriptors3D.PMI2(mol))
            rdk_dict['PMI3'].append(Descriptors3D.PMI3(mol))
            rdk_dict['Radius_of_Gyration'].append(Descriptors3D.RadiusOfGyration(mol))
            rdk_dict['Spherocity_Index'].append(Descriptors3D.SpherocityIndex(mol))
        rdk_df = pd.DataFrame.from_dict(rdk_dict)
        self.RDKitDescriptors = rdk_df


    def generate_MORDREDdescriptors(self):
        mord_dict = dict(PDB_File=[], Result=[])
        desc_3D = ['Mor01', 'Mor02', 'Mor03', 'Mor04', 'Mor05', 'Mor06', 'Mor07', 'Mor08', 'Mor09', 'Mor10', 'Mor11',
           'Mor12', 'Mor13', 'Mor14', 'Mor15', 'Mor16', 'Mor17', 'Mor18', 'Mor19', 'Mor20', 'Mor21', 'Mor22',
           'Mor23', 'Mor24', 'Mor25', 'Mor26', 'Mor27', 'Mor28', 'Mor29', 'Mor30', 'Mor31', 'Mor32', 'Mor01m',
           'Mor02m', 'Mor03m', 'Mor04m', 'Mor05m', 'Mor06m', 'Mor07m', 'Mor08m', 'Mor09m', 'Mor10m', 'Mor11m',
           'Mor12m', 'Mor13m', 'Mor14m', 'Mor15m', 'Mor16m', 'Mor17m', 'Mor18m', 'Mor19m', 'Mor20m', 'Mor21m',
           'Mor22m', 'Mor23m', 'Mor24m', 'Mor25m', 'Mor26m', 'Mor27m', 'Mor28m', 'Mor29m', 'Mor30m', 'Mor31m',
           'Mor32m', 'Mor01v', 'Mor02v', 'Mor03v', 'Mor04v', 'Mor05v', 'Mor06v', 'Mor07v', 'Mor08v', 'Mor09v',
           'Mor10v', 'Mor11v', 'Mor12v', 'Mor13v', 'Mor14v', 'Mor15v', 'Mor16v', 'Mor17v', 'Mor18v', 'Mor19v',
           'Mor20v', 'Mor21v', 'Mor22v', 'Mor23v', 'Mor24v', 'Mor25v', 'Mor26v', 'Mor27v', 'Mor28v', 'Mor29v',
           'Mor30v', 'Mor31v', 'Mor32v', 'Mor01se', 'Mor02se', 'Mor03se', 'Mor04se', 'Mor05se', 'Mor06se',
           'Mor07se', 'Mor08se', 'Mor09se', 'Mor10se', 'Mor11se', 'Mor12se', 'Mor13se', 'Mor14se', 'Mor15se',
           'Mor16se', 'Mor17se', 'Mor18se', 'Mor19se', 'Mor20se', 'Mor21se', 'Mor22se', 'Mor23se', 'Mor24se',
           'Mor25se', 'Mor26se', 'Mor27se', 'Mor28se', 'Mor29se', 'Mor30se', 'Mor31se', 'Mor32se', 'Mor01p',
           'Mor02p', 'Mor03p', 'Mor04p', 'Mor05p', 'Mor06p', 'Mor07p', 'Mor08p', 'Mor09p', 'Mor10p', 'Mor11p',
           'Mor12p', 'Mor13p', 'Mor14p', 'Mor15p', 'Mor16p', 'Mor17p', 'Mor18p', 'Mor19p', 'Mor20p', 'Mor21p',
           'Mor22p', 'Mor23p', 'Mor24p', 'Mor25p', 'Mor26p', 'Mor27p', 'Mor28p', 'Mor29p', 'Mor30p', 'Mor31p',
           'Mor32p', 'PNSA1', 'PNSA2', 'PNSA3', 'PNSA4', 'PNSA5', 'PPSA1', 'PPSA2', 'PPSA3', 'PPSA4', 'PPSA5',
           'DPSA1', 'DPSA2', 'DPSA3', 'DPSA4', 'DPSA5', 'FNSA1', 'FNSA2', 'FNSA3', 'FNSA4', 'FNSA5', 'FPSA1',
           'FPSA2', 'FPSA3', 'FPSA4', 'FPSA5', 'WNSA1', 'WNSA2', 'WNSA3', 'WNSA4', 'WNSA5', 'WPSA1', 'WPSA2',
           'WPSA3', 'WPSA4', 'WPSA5', 'RNCS', 'RPCS', 'TASA', 'TPSA', 'RASA', 'RPSA', 'GeomDiameter',
           'GeomRadius', 'GeomShapeIndex', 'GeomPetitjeanIndex', 'GRAV', 'GRAVH', 'GRAVp', 'GRAVHp', 'MOMI-X',
           'MOMI-Y', 'MOMI-Z']
    
        # Initialize calculator with specified descriptors
        calc = Calculator(descriptors, ignore_3D=False)
        calc.descriptors = [d for d in calc.descriptors if str(d) in desc_3D]

        results_df = pd.DataFrame()
        for item in self.d3_mol_dict:
            mol = self.d3_mol_dict[item]
            # Calculate descriptors
            result = calc.pandas([mol])
            # Add the name of the input file to the results DataFrame
            result.insert(0, "PDB_File", item)
            # Append the results to the main DataFrame
            results_df = results_df.append(result)
        self.MORDREDdescriptors = results_df

    def normalized_rdkit_descriptors(self):
        df = self.RDKitDescriptors
        # Extract the descriptors and convert them to a NumPy array
        X = df.iloc[:, 1:].to_numpy()
        # Select numeric columns (excluding the first column)
        numeric_columns = df.select_dtypes(include=np.number)
        numeric_columns.insert(0, "Molecule", df["PDB_File"])
        ## Normalisation of the 10 descriptors from RDKit using MinMaxScaler()
        # Create a copy of the numeric columns DataFrame
        normalized_df = numeric_columns.copy()
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        # Select only the numeric columns (excluding the first column)
        numeric_columns_only = numeric_columns.drop(columns='Molecule')
        # Normalize the descriptor values
        normalized_values = scaler.fit_transform(numeric_columns_only.values)
        # Update the values in the DataFrame with the normalized values
        normalized_df.loc[:, numeric_columns_only.columns] = normalized_values
        self.normalized_RDKitDescriptors = normalized_df
    
    def normalized_mordreddescriptors(self):
        df = self.MORDREDdescriptors
        # Select numeric columns (excluding the first column)
        numeric_columns = df.select_dtypes(include=np.number)
        numeric_columns.insert(0, "Molecule", df["PDB_File"])
        # Create a copy of the numeric columns DataFrame
        normalized_df = numeric_columns.copy()
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        # Select only the numeric columns (excluding the first column)
        numeric_columns_only = numeric_columns.drop(columns='Molecule')
        # Normalize the descriptor values
        normalized_values = scaler.fit_transform(numeric_columns_only.values)
        # Update the values in the DataFrame with the normalized values
        normalized_df.loc[:, numeric_columns_only.columns] = normalized_values
        self.normalized_MORDREDdescriptors = normalized_df


    
class GenerateFigures:
    '''
    Class to plot figures using 2D and 3D molecule structures
    '''
    def __init__(self, inputdata, d2_morgan_fp, d3_morgan_fp, normalized_RDKitDescriptors, normalized_MORDREDdescriptors, outdir):
        '''
        Initialize input file name
        '''
        self.fig_outdir = os.path.join(outdir, "figures")
        os.makedirs(self.fig_outdir, exist_ok=True)
        self.inputdata = inputdata
        self.d2_morgan_fp = d2_morgan_fp
        self.d3_morgan_fp = d3_morgan_fp
        self.normalized_RDKitDescriptors  = normalized_RDKitDescriptors 
        self.normalized_MORDREDdescriptors = normalized_MORDREDdescriptors

    def plot_3d(self):
        # Convert the column to a string
        d3_data = self.d3_morgan_fp
        d3_data['FP'] = d3_data['FP'].astype(str)
        d3_data = d3_data.rename(columns={'Name': 'UniqueMolecules'})
        fp_strings = d3_data['FP'].values
        molecules = d3_data['UniqueMolecules'].values
        # convert the fingerprint data to a numpy array
        fp_array = []
        for fp_string in fp_strings:
            fp_list = [float(x.strip()) for x in fp_string[1:-1].split(',')]
            fp_array.append(fp_list)

        fp_array = np.array(fp_array, dtype=object)

        # perform hierarchical clustering using the linkage function from scipy
        Z = linkage(fp_array, 'ward')

        # plot the dendrogram
        fig = plt.figure(figsize=(15, 10))
        dn = dendrogram(Z, labels=molecules, leaf_font_size=14, leaf_rotation=90)
        plt.title("Hierarchical Clustering Dendrogram of 390 3D Morgan Fingerprints")

        # Save the plot in the working directory as an image file (e.g., PNG)
        plt.savefig(os.path.join(self.fig_outdir, "3d_dendrogram.png"))  # Change the filename and format if needed (e.g., dendrogram.jpg)

        ## tSNE of 3D Morgan Fingerprints with labeling based on BGC Origin Group
        merged_3d_and_input = pd.merge(d3_data, self.inputdata, on='UniqueMolecules')
        labels = merged_3d_and_input['BGC_origin_group']
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
        X_tsne = tsne.fit_transform(fp_array)

        tab20 = cm.get_cmap('tab20')

        # Get unique BGC_origin_group values and assign colors to them
        unique_groups = labels.unique()
        color_map = {group: tab20.colors[i % len(tab20.colors)] for i, group in enumerate(unique_groups)}

        fig, ax = plt.subplots()
        for i, cluster in enumerate(sorted(labels.unique())):
            mask = labels == cluster
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=color_map[cluster], label=cluster, s=10)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')

        handles, labels = ax.get_legend_handles_labels()
        ncol = 1  # Number of labels per row
        nrows = int(np.ceil(len(labels) / ncol))
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), ncol=ncol)
        plt.title('t-SNE plot of 3D Morgan fingerprints: BGC Origin Group')
        # save the plot
        plt.savefig(os.path.join(self.fig_outdir, "3d_tSNE.png"))

    def plot_2d(self):
        ## tSNE of 2D Morgan Fingerprints with labeling based on BGC Origin Group
        merged_2d_and_input = pd.merge(self.d2_morgan_fp, self.inputdata, on='SMILES')
        merged_2d_and_input = merged_2d_and_input.drop_duplicates(subset=['UniqueMolecules'])
        merged_2d_and_input['Morgan_Fingerprint'] = merged_2d_and_input['Morgan_Fingerprint'].astype(str)
        fp_strings = merged_2d_and_input['Morgan_Fingerprint'].values
        # convert the fingerprint data to a numpy array
        fp_array = []
        for fp_string in fp_strings:
            fp_list = [float(x.strip()) for x in fp_string[1:-1].split(',') if x.strip()]
            fp_array.append(fp_list)

        fp_array = np.array(fp_array)

        labels = merged_2d_and_input['BGC_origin_group']

        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
        X_tsne = tsne.fit_transform(fp_array)

        tab20 = cm.get_cmap('tab20')

        # Get unique BGC_origin_group values and assign colors to them
        unique_groups = labels.unique()
        color_map = {group: tab20.colors[i % len(tab20.colors)] for i, group in enumerate(unique_groups)}

        fig, ax = plt.subplots()
        for i, cluster in enumerate(sorted(labels.unique())):
            mask = labels == cluster
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=color_map[cluster], label=cluster, s=10)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            
        handles, labels = ax.get_legend_handles_labels()
        ncol = 1  # Number of labels per row
        nrows = int(np.ceil(len(labels) / ncol))
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), ncol=ncol)
        plt.title('t-SNE plot of 2D Morgan fingerprints: BGC Origin Group')
        plt.savefig(os.path.join(self.fig_outdir, "2d_tSNE.png"))
    
    def plot_normalized_RDKitDescriptors(self):
        normalized_df = self.normalized_RDKitDescriptors
        # Select the desired columns
        selected_columns = normalized_df[['Asphericity', 'Eccentricity', 'Inertial_Shape_Factor', 'NPR1', 'NPR2', 'PMI2','PMI1', 'PMI3', 'Radius_of_Gyration', 'Spherocity_Index']]

        # Plot histograms for each column
        selected_columns.hist(bins=20, figsize=(16, 10), color='darkcyan')
        plt.suptitle('Distribution of Normalized Values: 10 RDKit 3D Descriptors ', fontsize=15)
        plt.grid(True)  # Remove the grid
        plt.savefig(os.path.join(self.fig_outdir, "normalized_RDKitDescriptors.png"))
 
    def plot_tsne_hc_RDKitDescriptors(self):
        normalized_df = self.normalized_RDKitDescriptors
        smiles_df = self.inputdata

        normalized_df['UniqueMolecules'] = normalized_df['Molecule'].apply(lambda x: x.replace('.pdb', ''))
        merged_df = pd.merge(normalized_df, smiles_df, on='UniqueMolecules')

        X = merged_df.loc[:, ['Asphericity', 'Eccentricity', 'Inertial_Shape_Factor', 'NPR1', 'NPR2', 'PMI2',
                            'PMI1', 'PMI3', 'Radius_of_Gyration', 'Spherocity_Index']].to_numpy()
        labels = merged_df['BGC_origin_group']

        tsne = TSNE(n_components=2, perplexity=30, random_state=0)
        X_tsne = tsne.fit_transform(X)
        
        tab20 = cm.get_cmap('tab20')

        # Get unique BGC_origin_group values and assign colors to them
        unique_groups = labels.unique()
        color_map = {group: tab20.colors[i % len(tab20.colors)] for i, group in enumerate(unique_groups)}

        fig, ax = plt.subplots()
        for i, cluster in enumerate(sorted(labels.unique())):
            mask = labels == cluster
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color_map[cluster], label=cluster, s=10)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')

        handles, labels = ax.get_legend_handles_labels()
        ncol = 1 # Number of labels per row
        nrows = int(np.ceil(len(labels) / ncol))
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), ncol=ncol)
        plt.title('tSNE of 10 normalized 3D descriptors: BGC Origin Group')
        plt.savefig(os.path.join(self.fig_outdir, "tSNE_RDKitDescriptors.png"))

        # Perform hierarchical clustering
        Z = linkage(X, method='ward')

        # Plot dendrogram
        plt.figure(figsize=(12, 6))
        plt.title('Hierarchical Clustering Dendrogram of Normalized 3D Descriptors for 390 Molecules')
        plt.xlabel('Unique Molecules')
        plt.ylabel('Distance')
        dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
        plt.savefig(os.path.join(self.fig_outdir, "HC_RDKitDescriptors.png"))

    def plot_normalized_MORDREDdescriptors(self):
        normalized_df = self.normalized_MORDREDdescriptors
        # Select the columns excluding the first column
        data_to_plot = normalized_df.iloc[:, 1:]
        # Plot the range and distribution of values for each column
        fig, axes = plt.subplots(nrows=11, ncols=5, figsize=(15, 25))

        for i, column in enumerate(data_to_plot.columns):
            ax = axes[i // 5, i % 5]
            data_to_plot[column].plot(ax=ax, kind='hist', bins=20, color='firebrick')
            ax.set_title(column)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_outdir, "normalized_MORDREDdescriptors.png"))
        
    def plot_tsne_hc_MORDREDdescriptors(self):
        normalized_df = self.normalized_MORDREDdescriptors
        smiles_df = self.inputdata

        normalized_df['UniqueMolecules'] = normalized_df['Molecule'].apply(lambda x: x.replace('.pdb', ''))
        
        merged_df = pd.merge(normalized_df, smiles_df, on='UniqueMolecules')

        # Remove the first and last elements
        X = merged_df.loc[:, normalized_df.columns.to_list()[1:-1]].to_numpy()
        labels = merged_df['BGC_origin_group']

        tsne = TSNE(n_components=2, perplexity=30, random_state=0)
        X_tsne = tsne.fit_transform(X)
        
        tab20 = cm.get_cmap('tab20')

        # Get unique BGC_origin_group values and assign colors to them
        unique_groups = labels.unique()
        color_map = {group: tab20.colors[i % len(tab20.colors)] for i, group in enumerate(unique_groups)}

        fig, ax = plt.subplots()
        for i, cluster in enumerate(sorted(labels.unique())):
            mask = labels == cluster
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color_map[cluster], label=cluster, s=10)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')

        handles, labels = ax.get_legend_handles_labels()
        ncol = 1 # Number of labels per row
        nrows = int(np.ceil(len(labels) / ncol))
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), ncol=ncol)
        plt.title('tSNE of normalized 3D MORDREDdescriptors: BGC Origin Group')
        plt.savefig(os.path.join(self.fig_outdir, "tSNE_MORDREDDescriptors.png"))


        # Perform hierarchical clustering
        Z = linkage(X, method='ward')

        # Plot dendrogram
        plt.figure(figsize=(12, 6))
        plt.title('Hierarchical Clustering Dendrogram of Normalized 3D MORDRED Descriptors')
        plt.xlabel('Unique Molecules')
        plt.ylabel('Distance')
        dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
        plt.savefig(os.path.join(self.fig_outdir, "HC_MORDREDDescriptors.png"))

        