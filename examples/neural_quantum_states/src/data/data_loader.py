import os
import numpy as np
import argparse
from src.data.read_hamiltonian import load_molecule
from tangelo.toolboxes.operators import count_qubits
from tangelo import Molecule

def load_hamiltonian_string(molecule_name: str, molecule_cid: int, basis: str, global_rank: int, world_size: int) -> [Molecule, int, str]:
    """
    Load the Hamiltonian string for a given molecule and basis set
    Args:
        molecule_name: name of molecule for Hamiltonian construction
        molecule_cid: Pubchem CID for the molecule (only needed/used if name not recognized)
        basis: name of basis set
        global_rank: identifying index among all GPUs
        world_size: number of GPUs
    Returns:
        molecule: a Tangelo Molecule object
        num_sites: number of qubits in Hamiltonian
        string: qubit Hamiltonian as a string
    """
    molecule, qubit_hamiltonian = load_molecule(molecule_name, molecule_cid, basis, global_rank, world_size)
    num_sites = count_qubits(qubit_hamiltonian)
    string = f"{qubit_hamiltonian}"
    return molecule, num_sites, string


def load_data(cfg: argparse.Namespace, global_rank: int, world_size: int) -> dict:
    """
    Load the data according to the given configuration (cfg).
    Args:
        cfg: config flags
        global_rank: indentifying index among all GPUs
        world_size: number of GPUs
    Returns:
        data: a dictionary containing the hamiltonian string, qubit/spin-orbital number, and molecule object
    """
    molecule_name = cfg.DATA.MOLECULE
    molecule_cid = cfg.DATA.MOLECULE_CID
    basis = cfg.DATA.BASIS
    molecule, num_sites, string = load_hamiltonian_string(molecule_name, molecule_cid, basis, global_rank, world_size)
    data = {'hamiltonian_string': string, 'num_sites': num_sites, 'molecule': molecule}
    return data
