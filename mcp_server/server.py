import argparse
import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

from dp.agent.server import CalculationMCPServer

from utils import SMILESListToCSV, UniMolPredict, filter_valid_smiles

# Load environment variables from .env file
load_dotenv()

# Directory of the fine-tuned UniMol perovskite weights (fixed_hyper_param),
# downloadable via download_weights.sh or from the GitHub Release
# `unimol-perovskite-v1` of this repository.
PEROVSKITE_WEIGHT_DIR = os.environ.get(
    "PEROVSKITE_WEIGHT_DIR", "/unimol_weight_perovskite/fixed_hyper_param"
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='UniMol MCP Server')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind to (default: 0.0.0.0)')
parser.add_argument('--port', type=int, default=50001, help='Port to listen on (default: 50001)')
args = parser.parse_args()

mcp = CalculationMCPServer("unimol_materials_tools", host=args.host, port=args.port)

@mcp.tool()
def Unimol_Predict_Perovskite_Additive(smiles_list: List[str]) -> Dict[str, float]:
    """
    Predict the additive effect of a perovskite PCE change with the additive.
    Args:
        smiles_list: List[str],
        smiles_list is a list of SMILES strings of the molecules to be predicted, do not include inorganic molecules, only include organic molecules.
    Returns:
        Dict[str, float]: Dictionary of smiles and predicted PCE change, the key is the SMILES string, the value is the predicted PCE change
    """

    filtered_smiles_list = filter_valid_smiles(smiles_list)
    if not filtered_smiles_list:
        print("Warning: No valid SMILES found in input list")
        return {}

    csv_path = SMILESListToCSV(filtered_smiles_list)
    unimol_results = UniMolPredict(Path(PEROVSKITE_WEIGHT_DIR), csv_path)
    #smiles_list and results should be a dictionary
    inductive_bias=0.2
    return {filtered_smiles_list[i]: unimol_results[i] + inductive_bias for i in range(len(filtered_smiles_list))}




if __name__ == "__main__":
    mcp.run(transport="streamable-http")
