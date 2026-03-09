pp-scattering-simulation
Toy Monte Carlo simulation of proton-proton scattering using Python
# Proton–Proton Elastic Scattering Simulation

This project implements a toy Monte Carlo simulation of elastic proton–proton scattering in the center-of-momentum (COM) frame using Python.

The simulation includes relativistic kinematics, Mandelstam variables, and a simple forward-peaked differential cross-section model.

## Physics Process

p + p → p + p

The scattering is simulated in the center-of-momentum frame where the initial momenta are equal and opposite.

## Features

- Relativistic center-of-momentum kinematics
- Mandelstam variables s, t, u
- Toy forward-peaked cross-section model
- Monte Carlo event generation
- Histogram visualization
- CSV output of generated events

## Physics Model

The differential cross section is modeled as

dσ/dt ∝ exp(B t)

which produces forward-peaked scattering typical of hadronic interactions.

## Run the simulation

```bash
python simulation.py
