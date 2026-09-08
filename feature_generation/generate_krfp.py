"""
KRFP (Klekota-Roth Fingerprint) Generator
==========================================

This script generates binary substructure fingerprints based on
Klekota-Roth SMARTS patterns for predicting perovskite solar cell
performance.

Reference:
Klekota, J. & Roth, F. P. (2008). Chemical substructures that
enrich for biological activity. Bioinformatics, 24(21), 2518-2525.
"""

import json
import numpy as np
import pandas as pd
from rdkit import Chem
from pathlib import Path


# Default KRFP SMARTS patterns
DEFAULT_KRFP_PATTERNS = {
    "primary_amine": "[NH2]",
    "secondary_amine": "[NH]([!H])[!H]",
    "tertiary_amine": "[N]([!H])([!H])[!H]",
    "hydroxyl": "[OH]",
    "carboxylic_acid": "[CX3](=O)[OX1H0-,OX2H1]",
    "ester": "[CX3](=O)[OX2H0]",
    "amide": "[NX3][CX3](=[OX1])",
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CH3][CX3H](=O)",
    "ketone": "[#6][CX3](=O)[#6]",
    "ether": "[OD2]([#6])[#6]",
    "thiol": "[SH]",
    "sulfide": "[#16X2]",
    "sulfoxide": "[SX3](=O)",
    "sulfone": "[SX4](=O)(=O)",
    "halogen": "[F,Cl,Br,I]",
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "iodine": "[I]",
    "aromatic_ring": "[a]",
    "benzene": "c1ccccc1",
    "pyridine": "n1ccccc1",
    "pyrimidine": "n1ccccn1",
    "pyrrole": "[nH]1cccc1",
    "furan": "o1cccc1",
    "thiophene": "s1cccc1",
    "imidazole": "n1c[nH]cc1",
    "triazole": "n1ncnc1",
    "pyrazole": "n1nccc1",
    "hetero_aromatic_6": "[n,o,s]1cccc1",
    "hetero_aromatic_5": "[n,o,s]1cccc1",
    "alkene": "[CX2]=[CX2]",
    "alkyne": "[CX1]#[CX2]",
    "nitrile": "[CX2]#[NX1]",
    "nitro": "[NX3](=[OX1])(=[OX1])",
    "azide": "[N-]=[N+]=[N-]",
    "phosphate": "[OX2]P(=[OX1])([OX2])([OX2])",
    "sulfonamide": "[SX4](=O)(=O)([NX3])",
    "guanidine": "[NH]C(=[NH])[NH2]",
    "urea": "[NH]C(=[O])[NH]",
    "carbamate": "[NX3][CX3](=[OX1])[OX2]",
    "carbonate": "[OX2]C(=[O])[OX2]",
    "imide": "[CX3](=[OX1])[NX3][CX3](=[OX1])",
    "aniline": "[NH2]c",
    "phenol": "[OH]c",
    "benzyl": "[CH2]c1ccccc1",
    "aryl_halide": "c[F,Cl,Br,I]",
    "aryl_nitrile": "c[CX2]#[NX1]",
    "aryl_amine": "c[NH,NH2,N]",
    "aryl_ether": "c[OX2]",
    "aryl_ketone": "c[CX3](=[OX1])",
    "heterocycle_3": "*1**1",
    "heterocycle_4": "*1***1",
    "heterocycle_5": "*1****1",
    "heterocycle_6": "*1*****1",
    "fused_aromatic": "c1ccc2c(c1)cccc2",
    "hydrogen_bond_donor": "[!#6;!H0]",
    "hydrogen_bond_acceptor": "[!#6;+0;!H0;!$([N,O,S][C,H])]",
    "rotatable_bond": "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]",
    "chiral_center": "[C@]",
    "sp2_carbon": "[CX3]",
    "sp3_carbon": "[CX4]",
    "sp_carbon": "[CX2]",
    "quaternary_carbon": "[C;X4;R0]",
    "terminal_methyl": "[CH3]",
    "methylene": "[CH2]",
    "methine": "[CH]",
}


class KRFPExtractor:
    """
    Extract Klekota-Roth fingerprints from SMILES strings.
    """

    def __init__(self, json_path=None, patterns=None):
        """
        Initialize KRFP extractor with SMARTS patterns.

        Parameters:
            json_path: Path to JSON file with SMARTS patterns
            patterns: Dictionary of pattern_name: SMARTS_string
        """
        if json_path is not None and Path(json_path).exists():
            with open(json_path, 'r') as f:
                self.smarts_dict = json.load(f)
        elif patterns is not None:
            self.smarts_dict = patterns
        else:
            self.smarts_dict = DEFAULT_KRFP_PATTERNS

        # Compile SMARTS patterns
        self.patterns = {}
        for key, smarts in self.smarts_dict.items():
            try:
                self.patterns[key] = Chem.MolFromSmarts(smarts)
                if self.patterns[key] is None:
                    print(f"Warning: Could not compile pattern '{key}': {smarts}")
            except Exception as e:
                print(f"Warning: Error compiling pattern '{key}': {e}")

        self.feature_names = list(self.patterns.keys())

    def get_krfp(self, smiles):
        """
        Generate binary fingerprint vector for a SMILES string.

        Parameters:
            smiles: SMILES string

        Returns:
            list: Binary fingerprint (1 = pattern present, 0 = absent)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * len(self.patterns)

        fingerprint = []
        for name, pattern in self.patterns.items():
            if pattern is not None:
                match = mol.HasSubstructMatch(pattern)
                fingerprint.append(int(match))
            else:
                fingerprint.append(0)

        return fingerprint

    def get_krfp_dict(self, smiles):
        """
        Generate fingerprint as dictionary with feature names.

        Parameters:
            smiles: SMILES string

        Returns:
            dict: Dictionary of feature_name: 0/1
        """
        fingerprint = self.get_krfp(smiles)
        return {name: value for name, value in zip(self.feature_names, fingerprint)}

    def transform(self, smiles_list):
        """
        Generate fingerprints for multiple SMILES.

        Parameters:
            smiles_list: List of SMILES strings

        Returns:
            np.ndarray: Binary fingerprint matrix (n_samples x n_features)
        """
        fingerprints = [self.get_krfp(smiles) for smiles in smiles_list]
        return np.array(fingerprints)

    def save_to_npy(self, smiles_list, output_path):
        """
        Generate and save fingerprints to .npy file.

        Parameters:
            smiles_list: List of SMILES strings
            output_path: Output file path
        """
        fingerprints = self.transform(smiles_list)
        np.save(output_path, fingerprints)
        print(f"Saved {len(fingerprints)} fingerprints to {output_path}")
        return fingerprints


def generate_krfp_for_csv(input_csv, output_path, smiles_col='SMILES'):
    """
    Generate KRFP features for molecules in a CSV file.

    Parameters:
        input_csv: Path to input CSV file
        output_path: Path to output .npy file
        smiles_col: Name of SMILES column

    Returns:
        np.ndarray: Fingerprint matrix
    """
    df = pd.read_csv(input_csv)
    smiles_list = df[smiles_col].tolist()

    extractor = KRFPExtractor()
    fingerprints = extractor.save_to_npy(smiles_list, output_path)

    return fingerprints


# ================================================================
# Main Execution
# ================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # Generate fingerprints from CSV file
        input_csv = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'krfp_features.npy'
        smiles_col = sys.argv[3] if len(sys.argv) > 3 else 'SMILES'

        print(f"Generating KRFP features from {input_csv}")
        fingerprints = generate_krfp_for_csv(input_csv, output_path, smiles_col)
        print(f"Shape: {fingerprints.shape}")
    else:
        # Demo
        print("KRFP Feature Generator Demo")
        print("=" * 50)

        sample_smiles = [
            'CC(=O)O',           # Acetic acid
            'c1ccccc1',          # Benzene
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',   # Caffeine
        ]

        extractor = KRFPExtractor()

        for smiles in sample_smiles:
            print(f"\nSMILES: {smiles}")
            fp_dict = extractor.get_krfp_dict(smiles)
            present = [k for k, v in fp_dict.items() if v == 1]
            print(f"Present patterns ({len(present)}): {', '.join(present[:10])}...")
