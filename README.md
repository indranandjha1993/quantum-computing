# Quantum Computing with Code

A hands-on, nine-notebook course in quantum computing. Every idea is introduced three ways:
as an **analogy**, as **maths**, and as **Qiskit code** you run, visualise and modify.
Circuits are drawn, states are plotted on Bloch spheres, and everything is simulated with
shot statistics on the Aer simulator, including a noisy simulated device in the final module.

## Course map

Start with `src/00_Start_Here.ipynb`: it checks your environment, introduces the four Qiskit
objects used throughout, and links to every module.

| # | Notebook | What you learn |
|---|----------|----------------|
| 0 | `00_Start_Here` | Setup check, the four Qiskit objects, conventions, cheat sheets, glossary |
| 1 | `01_The_Qubit_and_Superposition` | Amplitudes, Dirac notation, Bloch sphere, measurement, first circuit |
| 2 | `02_Quantum_Logic_Gates` | Gates as rotations, Pauli X/Y/Z, Hadamard, RY/RZ, circuits as matrix products |
| 3 | `03_Multiple_Qubits_and_Entanglement` | Tensor products, bit ordering, CNOT, Bell & GHZ states, a purity test for entanglement |
| 4 | `04_Measurement_and_Interference` | Born rule, shot noise, measurement bases, a quantum interferometer |
| 5 | `05_Quantum_Teleportation` | No-cloning, the protocol, deferred measurement vs real feed-forward, fidelity checks |
| 6 | `06_Deutsch_Jozsa_Algorithm` | Oracles, phase kickback, one query beats many, Bernstein-Vazirani |
| 7 | `07_Grovers_Search` | Amplitude amplification, oracle + diffuser, optimal iterations, scaling |
| 8 | `08_Noise_and_Real_Hardware` | Noise models, depth vs error, transpiling to a device, the repetition code |

Each module has learning objectives, "what to notice" callouts after every experiment,
interactive sliders where they help, exercises with hidden solutions, and key takeaways.

## Interactive labs

Eight simulators live in `src/qclab/` and are wired into the modules. Each needs a running kernel; when the
notebooks are viewed statically they show a preview of the initial state instead.

| Lab | Used in | What you do |
|-----|---------|-------------|
| Bloch Sphere Lab | Modules 1, 2 | Press gates and watch the arrow move along its rotation path. Shows amplitudes, angles, the running matrix, and fidelity to an optional target. |
| Circuit Composer | Start Here, Modules 2, 3 | Build circuits on 1 to 4 qubits from a gate menu. Live drawing, phase-coloured amplitudes, sampled histogram, per-qubit purity and Bloch arrows. Example circuits included. |
| Guess the State | Modules 1, 4 | A hidden qubit and a shot budget. Measure in Z, X or Y, then guess the state. Easy mode uses the six axis states, hard mode any point on the sphere. |
| Bell Test Lab | Module 3 | Choose a two-qubit state and four measurement angles. Live correlation curves and a CHSH gauge that only entangled states push past 2. |
| Teleportation Walkthrough | Module 5 | Step through the five stages with Bloch spheres, Alice's random bits, and Bob's fidelity at each step. |
| Grover Stepper | Module 7 | Step or auto-play through iterations: amplitude bars plus the rotating-arrow picture with both reflections drawn. |
| Noise Dial | Module 8 | Sliders for one-qubit, two-qubit and readout error. Ideal vs noisy histograms and a fidelity readout for Bell, GHZ, Grover, a CNOT chain, or your own composer circuit. |
| Checkpoint quiz | every module | Four questions per module with instant feedback and explanations. |

```python
from qclab import bloch_lab, circuit_composer, guess_the_state, quiz
from qclab import bell_test_lab, grover_stepper, teleport_walkthrough, noise_dial
bloch_lab(target='-')                      # reach |-> from |0>
circuit_composer(3, example='ghz')
guess_the_state('hard', budget=300)
quiz(4)
bell_test_lab()
grover_stepper(4, '0110')
teleport_walkthrough()
noise_dial()
```

The import works because Jupyter starts a notebook's kernel in the notebook's own folder. If you run a notebook
from elsewhere, add `src/` to `sys.path` first.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                           # creates .venv with every dependency
uv run python -m ipykernel install --user --name quantum_computing
jupyter lab src/                                                  # then pick the quantum_computing kernel
```

Notebooks ship with outputs, so you can read them on GitHub without running anything.
To run them yourself, open any module and use *Run All*.

## Stack

- **qiskit** 2.x for circuits, state vectors, operators and transpilation
- **qiskit-aer** for the `AerSimulator` backend and noise models
- **matplotlib** + **pylatexenc** + **sympy** for circuit diagrams, histograms, Bloch spheres and LaTeX states
- **ipywidgets** for the interactive sliders

## Editing the course

The notebooks are the source of truth: edit them directly in JupyterLab. If you add or reorder modules,
keep the navigation links at the top and bottom of each notebook pointing at the right neighbours and at
`00_Start_Here.ipynb`, and update the course map there and in this README.

## Related courses

This is one of three hands-on notebook courses built in the same format:

- [Data Structures and Algorithms](https://github.com/indranandjha1993/data-structures-algorithms): Big-O to dynamic programming, every claim measured
- [Design Patterns](https://github.com/indranandjha1993/design-patterns): the 23 Gang of Four patterns and the architectures they live in

## License

MIT. Use it, fork it, teach with it.
