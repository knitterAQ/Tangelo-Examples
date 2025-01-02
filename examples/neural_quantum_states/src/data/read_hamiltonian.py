import os
import time
import logging
import pickle
import pubchempy
import shelve
import torch
import torch.distributed as dist

from tangelo import Molecule, SecondQuantizedMolecule
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.operators import QubitOperator

def get_molecule_geometry(name: str, molecule_cid: int) -> list:
    '''
    Retrieves molecular geometry from PubChem
    Args:
        name: name of molecule
        molecule_cid: Pubchem CID of molecule (only used if name not recognized)
    Returns:
        geometry: a list of tuples describing the molecule's atomic nuclei and their spatial coordinates
    '''
    # PubChem IDs are stored in a shelf file - ID shelf is not comprehensive and may require updating (currently there is no automated process to add new molecule IDs)
    with shelve.open('./src/data/pubchem_IDs/ID_data') as MOLECULE_CID:
        if name in MOLECULE_CID.keys():
            id = MOLECULE_CID[name]
        else:
            # if name is not recognized, then program associates it with provided CID 
            id = molecule_cid
            MOLECULE_CID[name] = id
    geometry_flat = False
    pubchempy_molecule = pubchempy.get_compounds(id, 'cid', record_type='3d')
    if len(pubchempy_molecule) == 0:
        pubchempy_molecule = pubchempy.get_compounds(id, 'cid', record_type='2d')
        geometry_flat = True
    pubchempy_geometry = pubchempy_molecule[0].to_dict(properties=['atoms'])['atoms']
    if name == 'Cr2':
        pubchempy_geometry[0]['x'] = 0
        pubchempy_geometry[1]['x'] = 1.6788
    elif name == 'H2':
        pubchempy_geometry[0]['x'] = 0
        pubchempy_geometry[1]['x'] = 0.74
    elif name == 'LiH':
        pass
        pubchempy_geometry[0]['x'] = 0
        pubchempy_geometry[1]['x'] = 1.5949
    if not geometry_flat:
        geometry = [(atom['element'], (atom['x'], atom['y'], atom['z'])) for atom in pubchempy_geometry]
    else:
        geometry = [(atom['element'], (atom['x'], atom['y'], atom.get('z', 0))) for atom in pubchempy_geometry]
    return geometry

def load_molecule(molecule_name: str, molecule_cid: int, basis: str, global_rank: int, world_size: int) -> [Molecule, QubitOperator]:
    '''
    Retrieves second quantized molecule Hamiltonian for a given molecule and basis set
    Args:
        molecule_name: name of molecule
        molecule_cid: Pubchem CID of molecule (only used if name not recognized)
        basis: Basis set of molecule orbitals
        global_rank: identifying index among all GPUs
        world_size: number of GPUs
    Returns:
        molecule: Tangelo SecondQuantizedMolecule object
        qubit_hamiltonian: Second quantized molecular Hamiltonian as Tangelo QubitOperator object
    '''
    mult = 1 if molecule_name not in ["O2", "CH2"] else 3 #spin mulitiplicity
    if global_rank == 0:
        geometry = get_molecule_geometry(molecule_name, molecule_cid)
        num_atoms = [len(geometry)]
    else:
        num_atoms = [1]
    if world_size > 1:
        dist.broadcast_object_list(num_atoms,src=0)
    if global_rank > 0:
        geometry = [[] for _ in range(num_atoms[0])]
    if world_size > 1:
        dist.broadcast_object_list(geometry, src=0)

    #Solve molecule and print results
    logging.info("Creating Tangelo SecondQuantizedMolecule Object...")
    t_start=time.time()
    molecule = SecondQuantizedMolecule(geometry, q=0, spin=mult-1, basis=basis, frozen_orbitals=None) #RHF/ROHF used by default
    logging.info(f'{molecule_name} has:')
    logging.info(f'\t\t\tbasis {basis}')
    logging.info(f'\t\t\tgeometry {geometry},')
    logging.info(f'\t\t\t{molecule.n_active_electrons} electrons in {molecule.n_active_mos} spin-orbitals,')
    logging.info(f'\t\t\tHartree-Fock energy of {molecule.mf_energy:.6f} Hartree,')
    logging.info("done in {:.2f} seconds".format(time.time()-t_start))
    # 3. Convert molecular Hamiltonian to qubit Hamiltonian.
    logging.info("Obtain Qubit Hamiltonian... at {:.2f} seconds".format(time.time()-t_start))
    qubit_hamiltonian = fermion_to_qubit_mapping(molecule.fermionic_hamiltonian, "JW")
    logging.info("done in {:.2f} seconds".format(time.time()-t_start))
    return molecule, qubit_hamiltonian
