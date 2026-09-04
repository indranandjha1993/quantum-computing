"""Circuit Composer: build a small circuit and see its state, histogram and Bloch spheres."""
import numpy as np
import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit_aer import AerSimulator
from .common import phase_bars, show_in, is_static, caption, display_static

# name -> (QuantumCircuit method, number of qubits, takes an angle)
GATES = {
    "H": ("h", 1, False), "X": ("x", 1, False), "Y": ("y", 1, False), "Z": ("z", 1, False),
    "S": ("s", 1, False), "S-dag": ("sdg", 1, False), "T": ("t", 1, False), "T-dag": ("tdg", 1, False),
    "RX": ("rx", 1, True), "RY": ("ry", 1, True), "RZ": ("rz", 1, True), "P (phase)": ("p", 1, True),
    "CX (CNOT)": ("cx", 2, False), "CZ": ("cz", 2, False), "SWAP": ("swap", 2, False),
    "CCX (Toffoli)": ("ccx", 3, False),
}


def _ex_bell(qc):
    qc.h(0); qc.cx(0, 1)

def _ex_ghz(qc):
    qc.h(0)
    for t in range(1, qc.num_qubits):
        qc.cx(0, t)

def _ex_plus(qc):
    qc.h(range(qc.num_qubits))

def _ex_xhzh(qc):
    qc.x(0); qc.h(0); qc.z(0); qc.h(0)

def _ex_teleport(qc):
    qc.ry(1.2, 0)
    qc.h(1); qc.cx(1, 2)
    qc.cx(0, 1); qc.h(0)
    qc.cx(1, 2); qc.cz(0, 2)

def _ex_interf(qc):
    qc.h(0); qc.rz(np.pi / 2, 0); qc.h(0)

# name -> (required qubit count or None, builder)
EXAMPLES = {
    "(empty)": (None, None),
    "bell": (2, _ex_bell),
    "ghz": (None, _ex_ghz),
    "plus-all": (None, _ex_plus),
    "x-h-z-h": (1, _ex_xhzh),
    "interferometer": (1, _ex_interf),
    "teleport": (3, _ex_teleport),
}


class CircuitComposer:
    def __init__(self, n_qubits=2, example=None, shots=1000):
        self.sim = AerSimulator()
        self.n = int(n_qubits)
        self.qc = QuantumCircuit(self.n)
        self._build_ui(shots)
        if example:
            self.load_example(example)
        self._render()

    # ---- circuit edits -----------------------------------------------------
    def set_qubits(self, n):
        self.n = int(n)
        self.qc = QuantumCircuit(self.n)
        self._refresh_qubit_options()

    def load_example(self, name):
        need, builder = EXAMPLES[name]
        if need is not None and need != self.n:
            self.n_dd.value = need          # triggers set_qubits through the observer
        self.qc = QuantumCircuit(self.n)
        if builder is not None:
            builder(self.qc)

    def add_gate(self, name, qubits, angle=None):
        method, arity, has_angle = GATES[name]
        qubits = [int(q) for q in qubits[:arity]]
        if len(set(qubits)) != arity:
            raise ValueError("choose different qubits for control and target")
        if any(q >= self.n for q in qubits):
            raise ValueError("qubit index out of range")
        args = ([angle] if has_angle else []) + qubits
        getattr(self.qc, method)(*args)

    def undo(self):
        if len(self.qc.data):
            del self.qc.data[-1]

    def clear(self):
        self.qc = QuantumCircuit(self.n)

    # ---- views -------------------------------------------------------------
    def figures(self, shots):
        sv = Statevector(self.qc)
        fig_circ = self.qc.draw("mpl", scale=0.8)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
        phase_bars(sv, ax1)
        qm = self.qc.copy()
        qm.measure_all()
        counts = self.sim.run(qm, shots=shots).result().get_counts()
        plot_histogram(counts, ax=ax2, title=f"measurement, {shots} shots")
        fig.tight_layout()

        fig_bloch = plot_bloch_multivector(sv) if self.n <= 3 else None
        return fig_circ, fig, fig_bloch, sv

    def purity_html(self, sv):
        parts = []
        for i in range(self.n):
            others = [j for j in range(self.n) if j != i]
            p = float(partial_trace(sv, others).purity().real) if others else 1.0
            tag = " (entangled with the rest)" if p < 0.99 else ""
            parts.append(f"q{i}: purity {p:.2f}{tag}")
        note = "" if self.n <= 3 else " &nbsp; Bloch spheres are shown for up to 3 qubits."
        return "<div style='font-family:monospace; font-size:12px'>" + " &nbsp;|&nbsp; ".join(parts) + note + "</div>"

    # ---- widgets -----------------------------------------------------------
    def _build_ui(self, shots):
        self.n_dd = w.Dropdown(options=[1, 2, 3, 4], value=self.n, description="qubits", layout=w.Layout(width="140px"))
        self.example_dd = w.Dropdown(options=list(EXAMPLES), value="(empty)", description="example", layout=w.Layout(width="200px"))
        self.gate_dd = w.Dropdown(options=list(GATES), value="H", description="gate", layout=w.Layout(width="200px"))
        self.angle = w.IntSlider(value=90, min=-180, max=180, step=5, description="angle (deg)",
                                 continuous_update=False, layout=w.Layout(width="300px"))
        self.q_t = w.Dropdown(description="target", layout=w.Layout(width="140px"))
        self.q_c1 = w.Dropdown(description="control", layout=w.Layout(width="140px"))
        self.q_c2 = w.Dropdown(description="control 2", layout=w.Layout(width="150px"))
        self._refresh_qubit_options()
        self.shots = w.IntSlider(value=shots, min=100, max=4000, step=100, description="shots",
                                 continuous_update=False, layout=w.Layout(width="300px"))
        add = w.Button(description="Add gate", button_style="primary", layout=w.Layout(width="100px"))
        undo = w.Button(description="Undo", button_style="warning", layout=w.Layout(width="80px"))
        clear = w.Button(description="Clear", button_style="danger", layout=w.Layout(width="80px"))
        self.status = w.HTML()
        self.ket = w.HTMLMath()
        self.purity = w.HTML()
        self.circ_out = w.Output()
        self.state_out = w.Output()
        self.bloch_out = w.Output()

        self.n_dd.observe(lambda ch: self._on_n(ch["new"]), names="value")
        self.example_dd.observe(lambda ch: self._on_example(ch["new"]), names="value")
        self.gate_dd.observe(lambda ch: self._on_gate_choice(), names="value")
        self.shots.observe(lambda ch: self._render(), names="value")
        add.on_click(lambda _b: self._on_add())
        undo.on_click(lambda _b: (self.undo(), self._render()))
        clear.on_click(lambda _b: (self.clear(), self._render()))
        self._on_gate_choice()

        header = w.HTML("<b>Circuit Composer.</b> Pick a gate and qubits, press Add. "
                        "The state, a sampled histogram and each qubit's Bloch arrow update live.")
        self.ui = w.VBox([
            header,
            w.HBox([self.n_dd, self.example_dd, self.shots]),
            w.HBox([self.gate_dd, self.q_t, self.q_c1, self.q_c2, self.angle]),
            w.HBox([add, undo, clear, self.status]),
            self.circ_out, self.ket, self.state_out, self.purity, self.bloch_out,
        ])

    def _refresh_qubit_options(self):
        opts = list(range(self.n))
        for dd, default in ((self.q_t, 0), (self.q_c1, 1 if self.n > 1 else 0), (self.q_c2, 2 if self.n > 2 else 0)):
            dd.options = opts
            dd.value = default

    def _on_gate_choice(self):
        _m, arity, has_angle = GATES[self.gate_dd.value]
        self.angle.layout.display = None if has_angle else "none"
        self.q_c1.layout.display = None if arity >= 2 else "none"
        self.q_c2.layout.display = None if arity >= 3 else "none"

    def _on_n(self, n):
        self.set_qubits(n)
        self._render()

    def _on_example(self, name):
        self.load_example(name)
        self._render()

    def _on_add(self):
        try:
            name = self.gate_dd.value
            _m, arity, has_angle = GATES[name]
            if arity == 1:
                qubits = [self.q_t.value]
            elif arity == 2:
                qubits = [self.q_c1.value, self.q_t.value]
            else:
                qubits = [self.q_c1.value, self.q_c2.value, self.q_t.value]
            self.add_gate(name, qubits, np.radians(self.angle.value) if has_angle else None)
            self.status.value = ""
        except ValueError as e:
            self.status.value = f"<span style='color:#c00'>{e}</span>"
            return
        self._render()

    def _render(self):
        fig_circ, fig_state, fig_bloch, sv = self.figures(self.shots.value)
        show_in(self.circ_out, fig_circ)
        self.ket.value = "$$" + sv.draw("latex_source") + "$$"
        show_in(self.state_out, fig_state)
        self.purity.value = self.purity_html(sv)
        show_in(self.bloch_out, fig_bloch)


def circuit_composer(n_qubits=2, example=None, shots=1000, static=None):
    """Open the Circuit Composer. example: one of 'bell', 'ghz', 'plus-all', 'x-h-z-h', 'interferometer', 'teleport'."""
    comp = CircuitComposer(n_qubits, example, shots)
    if is_static(static):
        caption("Circuit Composer (static preview). Run this cell in JupyterLab for the interactive version.")
        fig_circ, fig_state, fig_bloch, sv = comp.figures(shots)
        display_static(fig_circ, fig_state, fig_bloch)
        display(HTML(comp.purity_html(sv)))
        return None
    display(comp.ui)
    return None
