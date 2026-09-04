"""Noise Dial: turn the error rates up and watch a circuit's answer degrade."""
import numpy as np
import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from . import composer as _composer
from .common import show_in, is_static, caption, display_static

BASIS = ["rz", "sx", "x", "cx"]


def make_noise_model(p1, p2, readout):
    nm = NoiseModel()
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x", "h", "u", "ry", "rx"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx", "cz"])
    if readout > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return nm


def _bell():
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); return qc

def _ghz():
    qc = QuantumCircuit(3); qc.h(0); qc.cx(0, 1); qc.cx(0, 2); return qc

def _teleport():
    qc = QuantumCircuit(3)
    qc.ry(1.2, 0); qc.h(1); qc.cx(1, 2); qc.cx(0, 1); qc.h(0); qc.cx(1, 2); qc.cz(0, 2)
    return qc

def _grover():
    from .grover import phase_oracle, diffuser
    qc = QuantumCircuit(3); qc.h(range(3))
    for _ in range(2):
        qc.compose(phase_oracle(3, "101"), inplace=True)
        qc.compose(diffuser(3), inplace=True)
    return qc

def _chain(depth):
    qc = QuantumCircuit(2)
    for _ in range(depth):
        qc.cx(0, 1)
        qc.barrier()
    return qc

CIRCUITS = {
    "Bell pair": _bell,
    "GHZ, 3 qubits": _ghz,
    "Teleportation (unitary form)": _teleport,
    "Grover n=3, 2 iterations": _grover,
    "CNOT chain (use the depth slider)": None,
    "composer (latest circuit)": None,
}


class NoiseDial:
    def __init__(self, circuit=None, shots=1000):
        self.custom = circuit
        self.sim_ideal = AerSimulator()
        self._build_ui(shots)
        self._render()

    def current_circuit(self):
        name = self.circuit_dd.value
        if name == "custom":
            return self.custom.copy()
        if name.startswith("CNOT chain"):
            return _chain(self.depth.value)
        if name.startswith("composer"):
            if _composer.LAST_CIRCUIT is None:
                return None
            return _composer.LAST_CIRCUIT.copy()
        return CIRCUITS[name]()

    def run(self, qc, p1, p2, readout, shots):
        # exact ideal distribution, noisy sampled distribution, transpiled depth and CNOT count
        ideal = {k: float(v) for k, v in Statevector(qc).probabilities_dict().items()}
        qm = qc.copy()
        qm.measure_all()
        sim = AerSimulator(noise_model=make_noise_model(p1, p2, readout))
        tqc = transpile(qm, basis_gates=BASIS, optimization_level=1)
        counts = sim.run(tqc, shots=shots).result().get_counts()
        F = float(hellinger_fidelity(ideal, counts))
        return ideal, counts, F, tqc.depth(), tqc.count_ops().get("cx", 0)

    def figure(self):
        qc = self.current_circuit()
        if qc is None:
            return None, "<span style='color:#c00'>Open the Circuit Composer first, then choose this option.</span>"
        p1, p2, ro, shots = self.p1.value / 100, self.p2.value / 100, self.ro.value / 100, self.shots.value
        ideal, counts, F, depth, ncx = self.run(qc, p1, p2, ro, shots)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
        expected = {k: round(v * shots) for k, v in ideal.items() if v > 1e-6}
        plot_histogram([expected, counts], legend=["ideal", "noisy"], ax=ax1, title="ideal vs noisy counts")
        if len(ideal) > 8:
            ax1.tick_params(axis="x", labelsize=7)

        if self.circuit_dd.value.startswith("CNOT chain"):
            depths = list(range(0, 121, 20))
            fids = [self.run(_chain(d), p1, p2, ro, 500)[2] for d in depths]
            ax2.plot(depths, fids, "o-", color="tab:red")
            ax2.axvline(self.depth.value, color="gray", ls="--", lw=1)
            ax2.set_xlabel("CNOT gates"); ax2.set_ylabel("fidelity with ideal")
            ax2.set_ylim(0, 1.05); ax2.grid(alpha=0.3)
            ax2.set_title("decay with depth at these error rates", fontsize=10)
        else:
            ax2.barh(["fidelity"], [F], color="tab:green" if F > 0.9 else ("tab:orange" if F > 0.7 else "tab:red"))
            ax2.set_xlim(0, 1); ax2.set_title("how close is the noisy answer to the ideal one?", fontsize=10)
            ax2.text(F, 0, f" {F:.3f}", va="center", fontsize=11)
        fig.tight_layout()
        html = (f"<span style='font-family:monospace; font-size:12px'>transpiled depth = {depth}, CNOT count = {ncx}, "
                f"Hellinger fidelity = {F:.3f} &nbsp; (1q error {p1:.1%}, 2q error {p2:.2%}, readout flip {ro:.1%})</span>")
        return fig, html

    # ---- widgets -----------------------------------------------------------
    def _build_ui(self, shots):
        options = list(CIRCUITS)
        if self.custom is not None:
            options = ["custom"] + options
        self.circuit_dd = w.Dropdown(options=options, value=options[0], description="circuit", layout=w.Layout(width="320px"))
        self.p1 = w.FloatSlider(value=0.1, min=0, max=5, step=0.1, description="1q error %", continuous_update=False, readout_format=".1f", layout=w.Layout(width="300px"))
        self.p2 = w.FloatSlider(value=2.0, min=0, max=10, step=0.25, description="2q error %", continuous_update=False, readout_format=".2f", layout=w.Layout(width="300px"))
        self.ro = w.FloatSlider(value=2.0, min=0, max=10, step=0.5, description="readout %", continuous_update=False, readout_format=".1f", layout=w.Layout(width="300px"))
        self.depth = w.IntSlider(value=20, min=0, max=120, step=10, description="chain depth", continuous_update=False, layout=w.Layout(width="300px"))
        self.shots = w.IntSlider(value=shots, min=200, max=4000, step=200, description="shots", continuous_update=False, layout=w.Layout(width="300px"))
        self.readout = w.HTML()
        self.out = w.Output()
        for widget in (self.circuit_dd, self.p1, self.p2, self.ro, self.depth, self.shots):
            widget.observe(lambda ch: self._render(), names="value")
        header = w.HTML("<b>Noise Dial.</b> Set the error rates of an imaginary device and see what it does to a circuit. "
                        "Fidelity 1 means the noisy histogram matches the ideal one.")
        self.ui = w.VBox([header, w.HBox([self.circuit_dd, self.shots]), w.HBox([self.p1, self.p2]),
                          w.HBox([self.ro, self.depth]), self.out, self.readout])

    def _render(self):
        fig, html = self.figure()
        show_in(self.out, fig)
        self.readout.value = html


def noise_dial(circuit=None, shots=1000, static=None):
    """Open the Noise Dial. Pass a QuantumCircuit (no measurements) to add it as the 'custom' option."""
    lab = NoiseDial(circuit, shots)
    if is_static(static):
        caption("Noise Dial (static preview). Run this cell in JupyterLab for the interactive version.")
        fig, html = lab.figure()
        display_static(fig)
        display(HTML(html))
        return None
    display(lab.ui)
    return None
