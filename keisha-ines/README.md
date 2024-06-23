![IQRG Banner for Research Projects](../IQRG_Banner_Research_Projects_2024.png)

# VQE for Ground State Optimization with Aqora

The group members of this project are Keisha Kwok and Ines Martin, mentored by Mr Jannes Stubbemann.
In this project, we use VQE (Variational Quantum Eigensolver) to find the ground state energy of the $H_2$ molecule using Pennylane.

## What is VQE?

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm designed to estimate the ground state energy of quantum systems. It uses the variational principle, which states that the expectation value of the Hamiltonian with any trial wavefunction is an upper bound to the ground state energy. The VQE algorithm works as follows:

1. Parameterization: A trial wavefunction (or quantum state) is prepared, parameterized by a set of variables. This state is typically represented by a quantum circuit that depends on these parameters.
2. Quantum Measurement: The Hamiltonian of the system (which encodes the energy information) is measured using the prepared state on a quantum computer.
3. Classical Optimization: The measured expectation value of the Hamiltonian is fed into a classical optimizer, which adjusts the parameters to minimize this value iteratively.
4. Ground State Approximation: The process repeats until the parameters converge, ideally yielding the ground state energy of the system.

VQE is particularly suited for noisy intermediate-scale quantum (NISQ) devices because it combines quantum measurements with classical optimization, thus reducing the required coherence time and making it feasible with current quantum hardware.

## Requirements

The libraries `PennyLane`, `matplotlib` and `xyz_parse` will be needed. Additionally, the library `aqora-cli` can also be installed

## Working with Aqora

If you want, ou can take part in the Aqora challenge to find the ground state energy of the $H_2$ molecule. You can follow the instructions [here](https://app.aqora.io/competitions/h2-groundstate-energy)
