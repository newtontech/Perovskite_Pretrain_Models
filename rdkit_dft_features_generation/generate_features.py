"""
Molecular Feature Generation for Perovskite Additives
======================================================

This script generates comprehensive molecular features using:
1. RDKit - for 2D molecular descriptors
2. Gaussian 16 + Multiwfn - for DFT electronic properties

Author: Generated for Perovskite_Pretrain_Models
"""

import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski


# ================================================================
# Part 1: RDKit 2D Descriptors
# ================================================================

def get_rdkit_descriptors(mol):
    """
    Calculate RDKit-based molecular descriptors.

    Parameters:
        mol: RDKit molecule object

    Returns:
        dict: Dictionary of descriptor names and values
    """
    desc = {}

    # Atom counts
    desc['C'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    desc['H'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'H')
    desc['N'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
    desc['F'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    desc['O'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')

    # Molecular properties
    desc['MW'] = Descriptors.MolWt(mol)                    # Molecular weight
    desc['LogP'] = Descriptors.MolLogP(mol)                # Hydrophobicity
    desc['TPSA'] = Descriptors.TPSA(mol)                   # Polar surface area
    desc['H_acceptor'] = Lipinski.NumHAcceptors(mol)       # H-bond acceptors
    desc['H_donor'] = Lipinski.NumHDonors(mol)             # H-bond donors
    desc['RB'] = Lipinski.NumRotatableBonds(mol)           # Rotatable bonds
    desc['Aromatic_rings'] = Descriptors.NumAromaticRings(mol)
    desc['Aliphatic_rings'] = Descriptors.NumAliphaticRings(mol)
    desc['Saturated_rings'] = Descriptors.NumSaturatedRings(mol)
    desc['Heteroatoms'] = Descriptors.NumHeteroatoms(mol)
    desc['QED'] = Descriptors.qed(mol)                     # Drug-likeness
    desc['IPC'] = Descriptors.Ipc(mol)                     # Complexity

    return desc


def smiles_to_3d(smiles, optimize=True):
    """
    Convert SMILES to 3D molecule structure.

    Parameters:
        smiles: SMILES string
        optimize: Whether to perform MMFF optimization

    Returns:
        RDKit molecule with 3D coordinates
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    if optimize:
        AllChem.MMFFOptimizeMolecule(mol)

    return mol


# ================================================================
# Part 2: Gaussian 16 Input File Generation
# ================================================================

def smiles_to_gjf(smiles, filename, charge=0, multiplicity=1,
                  method='B3LYP', basis='6-311+G(d,p)',
                  nproc=4, mem='8GB'):
    """
    Generate Gaussian input file (.gjf) from SMILES.

    Parameters:
        smiles: SMILES string
        filename: Output file path (.gjf)
        charge: Molecular charge
        multiplicity: Spin multiplicity
        method: DFT method (e.g., 'B3LYP', 'M06-2X')
        basis: Basis set (e.g., '6-311+G(d,p)', 'def2-SVP')
        nproc: Number of CPU cores
        mem: Memory allocation

    Returns:
        str: Content of the .gjf file
    """
    mol = smiles_to_3d(smiles)
    coords = mol.GetConformer().GetPositions()
    atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]

    content = f"%chk={filename.replace('.gjf', '')}.chk\n"
    content += f"%nproc={nproc}\n"
    content += f"%mem={mem}\n"
    content += f"#p {method}/{basis} opt freq\n\n"
    content += f"Generated from SMILES: {smiles}\n\n"
    content += f"{charge} {multiplicity}\n"

    for atom, coord in zip(atoms, coords):
        content += f"{atom} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n"

    content += "\n"

    return content


def run_gaussian(gjf_path, output_path=None):
    """
    Run Gaussian 16 calculation.

    Parameters:
        gjf_path: Path to input .gjf file
        output_path: Path to output .log file (optional)

    Returns:
        str: Path to output log file
    """
    if output_path is None:
        output_path = gjf_path.replace('.gjf', '.log')

    cmd = f"g16 < {gjf_path} > {output_path}"
    subprocess.run(cmd, shell=True, check=True)

    return output_path


def formchk(chk_path, fchk_path=None):
    """
    Convert .chk to .fchk file using formchk.

    Parameters:
        chk_path: Path to .chk file
        fchk_path: Path to output .fchk file (optional)

    Returns:
        str: Path to .fchk file
    """
    if fchk_path is None:
        fchk_path = chk_path.replace('.chk', '.fchk')

    cmd = f"formchk {chk_path} {fchk_path}"
    subprocess.run(cmd, shell=True, check=True)

    return fchk_path


# ================================================================
# Part 3: Multiwfn Analysis Functions
# ================================================================

def get_homo_lumo(fchk_path, multiwfn_cmd='Multiwfn_noGUI'):
    """
    Calculate HOMO and LUMO energies using Multiwfn.

    Parameters:
        fchk_path: Path to .fchk file
        multiwfn_cmd: Multiwfn command (default: Multiwfn_noGUI)

    Returns:
        tuple: (HOMO energy, LUMO energy) in eV
    """
    proc = subprocess.Popen(
        [multiwfn_cmd, fchk_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    commands = "0\n"  # View orbital information
    stdout, stderr = proc.communicate(commands)

    homo_energy = None
    lumo_energy = None

    for line in stdout.split('\n'):
        if "Orbital" in line and "HOMO" in line:
            parts = line.strip().split()
            homo_index = parts.index("HOMO,") + 4
            homo_energy = float(parts[homo_index])
        elif "Orbital" in line and "LUMO" in line:
            parts = line.strip().split()
            lumo_index = parts.index("LUMO,") + 4
            lumo_energy = float(parts[lumo_index])

    if homo_energy is None or lumo_energy is None:
        raise ValueError("Could not find HOMO or LUMO information")

    return homo_energy, lumo_energy


def get_total_dipole(fchk_path, multiwfn_cmd='Multiwfn_noGUI'):
    """
    Calculate total dipole moment using Multiwfn.

    Parameters:
        fchk_path: Path to .fchk file
        multiwfn_cmd: Multiwfn command

    Returns:
        float: Total dipole moment in Debye
    """
    proc = subprocess.Popen(
        [multiwfn_cmd, fchk_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    commands = "7\n4\nn\n"  # Main 7 -> Sub 4 -> Don't save charges
    stdout, stderr = proc.communicate(commands)

    dipole_moment = None
    for line in stdout.split('\n'):
        if "Total dipole moment from atomic charges" in line:
            parts = line.strip().split()
            dipole_au = float(parts[-2])
            dipole_moment = dipole_au * 2.5418  # Convert to Debye
            break

    if dipole_moment is None:
        raise ValueError("Could not find dipole moment information")

    return dipole_moment


def get_min_max_esp(fchk_path, isovalue=0.002, multiwfn_cmd='Multiwfn_noGUI'):
    """
    Calculate minimum and maximum ESP on molecular surface.

    Parameters:
        fchk_path: Path to .fchk file
        isovalue: Electron density isovalue for surface
        multiwfn_cmd: Multiwfn command

    Returns:
        tuple: (Min ESP, Max ESP) in eV
    """
    proc = subprocess.Popen(
        [multiwfn_cmd, fchk_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    commands = f"12\n1\n1\n{isovalue}\n0\n"
    stdout, stderr = proc.communicate(commands)

    min_esp = None
    max_esp = None

    for line in stdout.split('\n'):
        if "Global surface minimum" in line:
            parts = line.strip().split()
            min_esp = float(parts[3]) * 27.2114  # Convert a.u. to eV
        elif "Global surface maximum" in line:
            parts = line.strip().split()
            max_esp = float(parts[3])

    if min_esp is None or max_esp is None:
        raise ValueError("Could not find ESP information")

    return min_esp, max_esp


# ================================================================
# Part 4: Complete Feature Generation Pipeline
# ================================================================

def generate_all_features(smiles, work_dir='./tmp',
                          run_dft=True, multiwfn_cmd='Multiwfn_noGUI'):
    """
    Generate all molecular features for a SMILES string.

    Parameters:
        smiles: SMILES string
        work_dir: Working directory for calculations
        run_dft: Whether to run DFT calculations (requires Gaussian 16)
        multiwfn_cmd: Multiwfn command

    Returns:
        dict: Dictionary of all features
    """
    features = {'SMILES': smiles}

    # 1. Generate 3D structure and RDKit descriptors
    try:
        mol = smiles_to_3d(smiles)
        rdkit_features = get_rdkit_descriptors(mol)
        features.update(rdkit_features)
    except Exception as e:
        print(f"RDKit error: {e}")
        return features

    if not run_dft:
        return features

    # 2. Run Gaussian 16 calculation
    os.makedirs(work_dir, exist_ok=True)
    base_name = os.path.join(work_dir, 'molecule')

    try:
        # Generate .gjf file
        gjf_content = smiles_to_gjf(smiles, f'{base_name}.gjf')
        with open(f'{base_name}.gjf', 'w') as f:
            f.write(gjf_content)

        # Run Gaussian
        run_gaussian(f'{base_name}.gjf')

        # Convert to .fchk
        formchk(f'{base_name}.chk')

        # 3. Run Multiwfn analyses
        homo, lumo = get_homo_lumo(f'{base_name}.fchk', multiwfn_cmd)
        features['HOMO'] = homo
        features['LUMO'] = lumo
        features['Gap'] = lumo - homo

        min_esp, max_esp = get_min_max_esp(f'{base_name}.fchk', multiwfn_cmd=multiwfn_cmd)
        features['Min_ESP'] = min_esp
        features['Max_ESP'] = max_esp

        dipole = get_total_dipole(f'{base_name}.fchk', multiwfn_cmd)
        features['Dipole'] = dipole

    except Exception as e:
        print(f"DFT/Multiwfn error: {e}")

    return features


def batch_generate_features(smiles_list, output_csv='features.csv',
                            work_dir_base='./calc', run_dft=True):
    """
    Generate features for multiple molecules.

    Parameters:
        smiles_list: List of SMILES strings
        output_csv: Output CSV file path
        work_dir_base: Base directory for calculations
        run_dft: Whether to run DFT calculations

    Returns:
        pd.DataFrame: DataFrame with all features
    """
    all_features = []

    for i, smiles in enumerate(smiles_list):
        print(f"Processing {i+1}/{len(smiles_list)}: {smiles}")
        work_dir = os.path.join(work_dir_base, str(i))
        features = generate_all_features(smiles, work_dir, run_dft)
        features['index'] = i
        all_features.append(features)

    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)

    return df


# ================================================================
# Part 5: Main Execution
# ================================================================

if __name__ == '__main__':
    import sys

    # Example usage
    if len(sys.argv) > 1:
        # Single SMILES from command line
        smiles = sys.argv[1]
        features = generate_all_features(smiles, run_dft=False)
        print(pd.DataFrame([features]).to_string())
    else:
        # Demo with sample molecules
        sample_smiles = [
            'CC(=O)O',      # Acetic acid
            'c1ccccc1',     # Benzene
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'  # Ibuprofen
        ]

        print("Demo: Generating RDKit features (without DFT)")
        print("=" * 60)

        for smiles in sample_smiles:
            features = generate_all_features(smiles, run_dft=False)
            print(f"\nSMILES: {smiles}")
            for key, value in features.items():
                if key != 'SMILES':
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
