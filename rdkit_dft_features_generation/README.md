# Molecular Feature Generation for Perovskite Additives

This folder contains documentation and sample code for generating molecular features used in predicting perovskite solar cell performance. The features are generated using the following tools:

1. **RDKit** - Molecular descriptors and fingerprints
2. **Gaussian 16** - DFT calculations for electronic properties
3. **Multiwfn** - Wavefunction analysis for ESP and orbital properties
4. **CP2K** - (Optional) DFT calculations for large systems
5. **VESTA** - Crystal structure visualization
6. **Pymatgen** - Materials analysis

## Feature Table Overview

| Feature | Tool | Method | Unit | Description |
|---------|------|--------|------|-------------|
| **Atom Counts** |
| C | RDKit | `mol.GetAtoms()` | count | Number of Carbon atoms |
| H | RDKit | `mol.GetAtoms()` | count | Number of Hydrogen atoms |
| N | RDKit | `mol.GetAtoms()` | count | Number of Nitrogen atoms |
| F | RDKit | `mol.GetAtoms()` | count | Number of Fluorine atoms |
| O | RDKit | `mol.GetAtoms()` | count | Number of Oxygen atoms |
| **Molecular Descriptors** |
| MW | RDKit | `Descriptors.MolWt()` | g/mol | Molecular weight |
| LogP | RDKit | `Descriptors.MolLogP()` | - | Partition coefficient (hydrophobicity) |
| TPSA | RDKit | `Descriptors.TPSA()` | Ų | Topological polar surface area |
| H_acceptor | RDKit | `Lipinski.NumHAcceptors()` | count | Number of hydrogen bond acceptors |
| H_donor | RDKit | `Lipinski.NumHDonors()` | count | Number of hydrogen bond donors |
| RB | RDKit | `Lipinski.NumRotatableBonds()` | count | Number of rotatable bonds |
| Aromatic_rings | RDKit | `Descriptors.NumAromaticRings()` | count | Number of aromatic rings |
| Aliphatic_rings | RDKit | `Descriptors.NumAliphaticRings()` | count | Number of aliphatic rings |
| Saturated_rings | RDKit | `Descriptors.NumSaturatedRings()` | count | Number of saturated rings |
| Heteroatoms | RDKit | `Descriptors.NumHeteroatoms()` | count | Number of heteroatoms |
| QED | RDKit | `Descriptors.qed()` | 0-1 | Drug-likeness score |
| IPC | RDKit | `Descriptors.Ipc()` | - | Molecular complexity index |
| **Electronic Properties (DFT)** |
| HOMO | Gaussian 16 + Multiwfn | B3LYP/6-311+G(d,p) | eV | Highest occupied molecular orbital energy |
| LUMO | Gaussian 16 + Multiwfn | B3LYP/6-311+G(d,p) | eV | Lowest unoccupied molecular orbital energy |
| Gap | Calculation | LUMO - HOMO | eV | HOMO-LUMO energy gap |
| Min_ESP | Gaussian 16 + Multiwfn | ESP on 0.002 isosurface | eV | Minimum electrostatic potential |
| Max_ESP | Gaussian 16 + Multiwfn | ESP on 0.002 isosurface | eV | Maximum electrostatic potential |
| Dipole | Gaussian 16 + Multiwfn | Atomic charge method | Debye | Total dipole moment |

## Workflow Overview

```
SMILES → RDKit (2D descriptors) → Gaussian 16 (DFT) → Multiwfn (Analysis) → Feature Vector
   │            │                       │                     │
   │            │                       │                     │
   ▼            ▼                       ▼                     ▼
 3D coords   MW, LogP,              HOMO, LUMO            ESP, Dipole
 conformer   TPSA, etc.             opt+freq              analysis
```

## Detailed Method Descriptions

### 1. RDKit Features (2D Descriptors)

**Installation:**
```bash
conda install -c conda-forge rdkit
```

**Key Methods:**
- `Chem.MolFromSmiles(smiles)` - Parse SMILES string
- `Chem.AddHs(mol)` - Add explicit hydrogens
- `AllChem.EmbedMolecule(mol)` - Generate 3D coordinates
- `AllChem.MMFFOptimizeMolecule(mol)` - Force field optimization

### 2. Gaussian 16 Calculations

**Input File Format (.gjf):**
```
%chk=molecule.chk
%nproc=4
%mem=8GB
#p B3LYP/6-311+G(d,p) opt freq

Generated from SMILES: [SMILES string]

0 1
C  x1 y1 z1
H  x2 y2 z2
...
```

**Key Commands:**
- `opt` - Geometry optimization
- `freq` - Frequency calculation (confirms minimum)
- `B3LYP/6-311+G(d,p)` - DFT functional and basis set

### 3. Multiwfn Analysis

**Installation:**
```bash
# Download from http://sobereva.com/multiwfn/
# Set environment variable
export Multiwfnpath=/path/to/Multiwfn
```

**Key Functions:**
- Function 0: View orbital information (HOMO/LUMO)
- Function 7,4: Calculate dipole moment
- Function 12,1: ESP analysis on molecular surface

### 4. Feature Values Reference

Based on the perovskite additive dataset (447 molecules):

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| C | 0 | 48 | 8.4 | 6.2 |
| H | 0 | 50 | 10.1 | 7.8 |
| MW | 60.05 | 991.16 | 214.3 | 145.2 |
| LogP | -2.67 | 12.78 | 1.82 | 2.15 |
| TPSA | 0.0 | 223.8 | 52.3 | 38.4 |
| HOMO (eV) | -9.32 | -4.36 | -6.72 | 0.89 |
| LUMO (eV) | -5.00 | 0.63 | -1.48 | 1.12 |
| Gap (eV) | 0.98 | 8.27 | 5.24 | 1.21 |
| Dipole (D) | 0.0 | 313.9 | 24.8 | 42.1 |

## Sample Output

```csv
,SMILES,C,H,N,F,O,MW,LogP,TPSA,H_acceptor,H_donor,RB,Aromatic_rings,Aliphatic_rings,Saturated_rings,Heteroatoms,QED,IPC,HOMO,LUMO,Gap,Min_ESP,Max_ESP,Dipole
0,CC1COC(=O)O1,4,0,0,0,3,102.09,0.54,35.53,3,0,0,0,1,1,3,0.42,46.67,-8.31,-0.04,8.27,-2.17,1.89,7.03
1,[B-](F)(F)(F)F.CCCCN1C=C[N+](=C1)C,8,0,2,4,0,226.03,2.41,8.81,1,0,3,1,0,0,7,0.42,1357.33,-7.52,-1.21,6.31,-2.16,2.36,8.80
```

## References

1. RDKit: Open-source cheminformatics. http://www.rdkit.org
2. Gaussian 16, Revision C.01, M. J. Frisch et al., Gaussian, Inc., Wallingford CT, 2016.
3. Multiwfn: T. Lu, F. Chen, J. Comput. Chem. 2012, 33, 580-592.
4. CP2K: https://www.cp2k.org/
5. VESTA: K. Momma and F. Izumi, J. Appl. Crystallogr. 2011, 44, 1272-1276.
6. Pymatgen: S. P. Ong et al., Comput. Mater. Sci. 2013, 68, 314-319.
7. KRFP: Klekota-Roth Fingerprints for substructure matching
