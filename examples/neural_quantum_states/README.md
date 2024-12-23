# AUTOREGRESSIVE NEURAL QUANTUM STATES FOR QUANTUM CHEMISTRY

This repository contains code jointly developed between the University of Michigan and SandboxAQ to implement the retentive network (RetNet) neural quantum states ansatz outlined in the paper, "Retentive Neural Quantum States: Efficient Ansatze for Ab Initio Quantum Chemistry," by Oliver Knitter, Dan Zhao, James Stokes, Martin Ganahl, Stefan Leichenauer, and Shravan Veerapaneni.

Preprint available on the arXiv: https://arxiv.org/abs/2411.03900

Corresponding Author: Oliver Knitter, knitter@umich.edu

This repository is based off of the code Tianchen Zhao released (https://github.com/Ericolony/made-qchem) alongside the paper "Scalable neural quantum states architecture for quantum chemistry" (https://arxiv.org/abs/2208.05637). This new code uses neural quantum states (NQS), implemented in PyTorch to calculate electronic ground state energies for second quantized molecular Hamiltonians. Though not exactly a quantum or hybrid algorithm, this workflow makes use of Tangelo (https://github.com/sandbox-quantum/Tangelo/) and PySCF to calculate Jordan--Wigner encodings of the electronic Hamiltonians, along with some classical chemistry benchmark values. These uses of Tangelo are mainly found in the subdirectory src/data/.

The RetNet ansatz implementation was made using the yet-another-retnet repository (https://github.com/fkodom/yet-another-retnet). Other ansatze available are the MADE and Transformer ansatze, which are implemented natively in PyTorch. The Hamiltonian expectation value estimates are calculated following the procedure outlined in Zhao et al.'s paper,  but the modular structure of the code allows for relatively simple plug-and-play implementations of different models and Hamiltonian expectation estimate calculators.

## Usage

### 1. Environment Setup

- The environment requirements are available in the nqs-qchem.yml file.

- Make sure your operating system meets the requirements for PyTorch 2.5.1 and CUDA 11.8.

- If setting up the environment through the .yml file does not work, then the primary dependencies are obtained as follows:

    ```
    conda install python==3.9.18
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install tensorboardX
    conda install pytorch-model-summary
    conda install yacs
    pip install yet-another-retnet
    pip install tangelo-gc
    pip install --prefer-binary pyscf
    ```

### 2. Running the Program

To view a list of previously logged molecule names, without running the main program, run

```
python -m main --list_molecules
```

If a desired molecule name is not stored by the program, then running the program with the desired name and a corresponding pubchem ID will link the name and ID in the program's internal name storage. The list of orbital basis sets is available within the PySCF documentation. The program itself can be run by executing the following script:

```
./run.sh
```

The run script will adapt to the number of GPUs specified for use, but it does so in a naive way. The script presumes that the total number of GPUs used by the program equals the total number of system GPUs available on the current machine, and will call on all those GPUs to run the program. It is the user's responsibility to ensure these parameters are adequate, and to modify them if necessary (to run the program using multiple nodes, for instance).
Input configuration flags are generally recorded in .yaml files within the directory ./exp_configs/ and are called as needed in the run script. Some configurations are expected to be specified within the run script: hypothetically all configurations may be specified in the run script, which supersedes the .yaml file, but this is not the expected form of use.

### 3. Logging Outputs
All logged results and outputs produced by the program can be found under './logger/'. Each run of the program updates './logger/results.txt' (which is not automatically erased) with the best result obtained over the specified number of trials. Each run of the program generates its own output directory within './logger/' that stores a copy of the .yaml file containing the input configurations that were used (those not specified by the file are included in the name of the directory). A dictionary containing basic logging information for all trials is saved as 'result.npy', and tensorboard files containing identical information are saved as well. The model corresponding with the best score of all the trials is also saved here. It is possible to specify different save locations for all these data using existing input configurations.
