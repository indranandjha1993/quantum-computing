"""Teleportation Walkthrough: the protocol one stage at a time, with Alice's random outcome."""
import numpy as np
import ipywidgets as w
from IPython.display import display, HTML
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity
from qiskit.visualization import plot_bloch_multivector
from .common import state_from_angles, show_in, is_static, caption, display_static

STAGES = [
    ("0. Start",
     "Alice's payload |psi> sits on qubit 0. Qubits 1 and 2 are fresh |0> states. Bob's qubit (2) knows nothing yet."),
    ("1. Entangled channel",
     "H on qubit 1 and CNOT(1, 2) make a Bell pair. Alice keeps qubit 1, Bob takes qubit 2. Their arrows vanish: each half is maximally mixed on its own."),
    ("2. Alice's interaction",
     "CNOT(0, 1) then H on qubit 0. This is a Bell-basis measurement in disguise: the payload is now spread across all three qubits and none of them holds |psi> on its own. "
     "Bob's arrow is still zero, and the fidelity of his qubit with |psi> is exactly 1/2, pure chance."),
    ("3. Alice measures",
     "Alice measures qubits 0 and 1 and gets two random bits (each of the four outcomes has probability 1/4, whatever |psi> is). The state collapses. "
     "Look at Bob's qubit: it is already |psi> up to one of four twists, selected by her bits."),
    ("4. Classical message",
     "Alice sends the two bits to Bob over an ordinary channel. Nothing quantum travels. Until they arrive, Bob's qubit is useless to him."),
    ("5. Bob corrects",
     "Bob applies X if the qubit-1 bit is 1 and Z if the qubit-0 bit is 1. His qubit is now exactly |psi>. Alice's copy was destroyed at stage 3."),
]


def project(sv, m0, m1):
    # Collapse a 3-qubit state onto qubit 0 = m0 and qubit 1 = m1, then renormalise.
    amps = np.array(sv.data, dtype=complex)
    for i in range(len(amps)):
        if ((i >> 0) & 1) != m0 or ((i >> 1) & 1) != m1:
            amps[i] = 0
    norm = np.linalg.norm(amps)
    return Statevector(amps / norm)


class TeleportWalkthrough:
    def __init__(self, theta=1.4, phi=1.0, seed=None):
        self.rng = np.random.default_rng(seed)
        self.theta, self.phi = float(theta), float(phi)
        self.remeasure()
        self._build_ui()
        self._render()

    def remeasure(self):
        outcome = int(self.rng.integers(4))
        self.m0, self.m1 = outcome & 1, (outcome >> 1) & 1

    def payload(self):
        return state_from_angles(self.theta, self.phi)

    def stage_circuit(self, stage):
        qc = QuantumCircuit(3)
        qc.u(self.theta, self.phi, 0, 0)
        if stage >= 1:
            qc.barrier(); qc.h(1); qc.cx(1, 2)
        if stage >= 2:
            qc.barrier(); qc.cx(0, 1); qc.h(0)
        return qc

    def stage_state(self, stage):
        sv = Statevector(self.stage_circuit(stage))
        if stage >= 3:
            sv = project(sv, self.m0, self.m1)
        if stage >= 5:
            fix = QuantumCircuit(3)
            if self.m1:
                fix.x(2)
            if self.m0:
                fix.z(2)
            sv = sv.evolve(fix)
        return sv

    def drawn_circuit(self, stage):
        qc = self.stage_circuit(stage)
        if stage >= 3:
            qc.barrier()
            qc.add_register(ClassicalRegister(2, "alice"))
            qc.measure(0, 0); qc.measure(1, 1)
        if stage >= 5:
            qc.barrier()
            if self.m1:
                qc.x(2)
            if self.m0:
                qc.z(2)
        return qc

    def bob_fidelity(self, sv):
        return float(state_fidelity(partial_trace(sv, [0, 1]), self.payload()))

    def info_html(self, stage):
        title, text = STAGES[stage]
        sv = self.stage_state(stage)
        F = self.bob_fidelity(sv)
        bits = f"qubit 0 bit = {self.m0}, qubit 1 bit = {self.m1}" if stage >= 3 else "not measured yet"
        extra = ""
        if stage in (3, 4):
            twist = {(0, 0): "none, Bob already holds |psi>", (0, 1): "X", (1, 0): "Z", (1, 1): "X then Z"}[(self.m0, self.m1)]
            extra = f"<br><b>Twist on Bob's qubit for this outcome:</b> {twist}"
        return (f"<div style='font-size:13px; line-height:1.6'><b>{title}</b><br>{text}<br>"
                f"<b>Alice's bits:</b> {bits}{extra}<br>"
                f"<b>Fidelity of Bob's qubit with |psi>:</b> {F:.3f}</div>")

    # ---- widgets -----------------------------------------------------------
    def _build_ui(self):
        self.theta_s = w.IntSlider(value=int(round(np.degrees(self.theta))), min=0, max=180, step=5, description="theta (deg)", continuous_update=False, layout=w.Layout(width="300px"))
        self.phi_s = w.IntSlider(value=int(round(np.degrees(self.phi))), min=0, max=355, step=5, description="phi (deg)", continuous_update=False, layout=w.Layout(width="300px"))
        self.stage = w.IntSlider(value=0, min=0, max=5, step=1, description="stage", continuous_update=False, layout=w.Layout(width="300px"))
        prev = w.Button(description="Prev", layout=w.Layout(width="70px"))
        nxt = w.Button(description="Next", button_style="primary", layout=w.Layout(width="70px"))
        remeasure = w.Button(description="Re-measure", layout=w.Layout(width="110px"))
        self.info = w.HTML()
        self.circ_out = w.Output()
        self.bloch_out = w.Output()

        self.theta_s.observe(lambda ch: self._on_payload(), names="value")
        self.phi_s.observe(lambda ch: self._on_payload(), names="value")
        self.stage.observe(lambda ch: self._render(), names="value")
        prev.on_click(lambda _b: setattr(self.stage, "value", max(0, self.stage.value - 1)))
        nxt.on_click(lambda _b: setattr(self.stage, "value", min(5, self.stage.value + 1)))
        remeasure.on_click(lambda _b: (self.remeasure(), self._render()))

        header = w.HTML("<b>Teleportation Walkthrough.</b> Choose Alice's payload, then step through the protocol. "
                        "Re-measure draws a different random outcome for Alice's bits.")
        self.ui = w.VBox([header, w.HBox([self.theta_s, self.phi_s]), w.HBox([prev, nxt, self.stage, remeasure]),
                          self.info, self.circ_out, self.bloch_out])

    def _on_payload(self):
        self.theta, self.phi = np.radians(self.theta_s.value), np.radians(self.phi_s.value)
        self._render()

    def _render(self):
        s = self.stage.value
        self.info.value = self.info_html(s)
        show_in(self.circ_out, self.drawn_circuit(s).draw("mpl", scale=0.75))
        show_in(self.bloch_out, plot_bloch_multivector(self.stage_state(s)))


def teleport_walkthrough(theta=1.4, phi=1.0, seed=None, static=None):
    """Open the Teleportation Walkthrough with a payload at Bloch angles (theta, phi) in radians."""
    lab = TeleportWalkthrough(theta, phi, seed)
    if is_static(static):
        caption("Teleportation Walkthrough (static preview of stage 3). Run this cell in JupyterLab for the interactive version.")
        display(HTML(lab.info_html(3)))
        display_static(lab.drawn_circuit(3).draw("mpl", scale=0.75), plot_bloch_multivector(lab.stage_state(3)))
        return None
    display(lab.ui)
    return None
