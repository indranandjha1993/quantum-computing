"""Bell Test Lab: correlations, measurement angles and the CHSH inequality."""
import numpy as np
import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .common import show_in, is_static, caption, display_static

CLASSICAL_BOUND = 2.0
QUANTUM_BOUND = 2 * np.sqrt(2)


def _plus_plus(qc):
    qc.h([0, 1])

def _zero_zero(qc):
    pass

def _phi_plus(qc):
    qc.h(0); qc.cx(0, 1)

def _phi_minus(qc):
    qc.x(0); qc.h(0); qc.cx(0, 1)

def _psi_plus(qc):
    qc.x(1); qc.h(0); qc.cx(0, 1)

def _psi_minus(qc):
    qc.x(0); qc.x(1); qc.h(0); qc.cx(0, 1)

STATES = {
    "Bell Phi+  (|00> + |11>)": _phi_plus,
    "Bell Phi-  (|00> - |11>)": _phi_minus,
    "Bell Psi+  (|01> + |10>)": _psi_plus,
    "Bell Psi-  (|01> - |10>)": _psi_minus,
    "product |+>|+>": _plus_plus,
    "product |0>|0>": _zero_zero,
}

PRESETS = {
    "CHSH optimal (0, 90, 45, 135)": (0, 90, 45, 135),
    "aligned (0, 90, 0, 90)": (0, 90, 0, 90),
}


def make_state(name):
    qc = QuantumCircuit(2)
    STATES[name](qc)
    return Statevector(qc)


def correlation(sv, a, b, shots=None):
    # Alice (qubit 0) measures along angle a in the X-Z plane, Bob (qubit 1) along b.
    # RY(-angle) rotates that axis onto Z, then a standard measurement gives +1 for 0 and -1 for 1.
    qc = QuantumCircuit(2)
    qc.ry(-a, 0)
    qc.ry(-b, 1)
    rot = sv.evolve(qc)
    sign = lambda k: 1 if k[0] == k[1] else -1
    exact = float(sum(p * sign(k) for k, p in rot.probabilities_dict().items()))
    if shots is None:
        return exact, None
    counts = rot.sample_counts(shots)
    sampled = float(sum(int(c) * sign(k) for k, c in counts.items()) / shots)
    return exact, sampled


def chsh(sv, a, a2, b, b2, shots=None):
    pairs = {"E(a,b)": (a, b), "E(a,b')": (a, b2), "E(a',b)": (a2, b), "E(a',b')": (a2, b2)}
    vals = {k: correlation(sv, *ang, shots=shots) for k, ang in pairs.items()}
    S_exact = vals["E(a,b)"][0] - vals["E(a,b')"][0] + vals["E(a',b)"][0] + vals["E(a',b')"][0]
    S_sampled = None
    if shots is not None:
        S_sampled = vals["E(a,b)"][1] - vals["E(a,b')"][1] + vals["E(a',b)"][1] + vals["E(a',b')"][1]
    return vals, S_exact, S_sampled


class BellTestLab:
    def __init__(self, state=None, shots=1000):
        self.state_name = state or list(STATES)[0]
        self._build_ui(shots)
        self._render()

    # ---- computation -------------------------------------------------------
    def angles(self):
        return tuple(np.radians(s.value) for s in (self.a, self.a2, self.b, self.b2))

    def figure(self):
        sv = make_state(self.state_dd.value)
        a, a2, b, b2 = self.angles()
        shots = self.shots.value
        vals, S_exact, S_sampled = chsh(sv, a, a2, b, b2, shots)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
        bs = np.linspace(0, np.pi, 91)
        for ang, label, color in ((a, "Alice at a", "tab:blue"), (a2, "Alice at a'", "tab:purple")):
            ax1.plot(np.degrees(bs), [correlation(sv, ang, x)[0] for x in bs], color=color, label=label)
        pts = [(b, vals["E(a,b)"][1], "tab:blue"), (b2, vals["E(a,b')"][1], "tab:blue"),
               (b, vals["E(a',b)"][1], "tab:purple"), (b2, vals["E(a',b')"][1], "tab:purple")]
        for x, y, color in pts:
            ax1.plot(np.degrees(x), y, "o", color=color, markersize=8, markeredgecolor="k")
        ax1.axhline(0, color="k", lw=0.6)
        ax1.set_ylim(-1.15, 1.15); ax1.set_xlim(0, 180)
        ax1.set_xlabel("Bob's angle (deg)"); ax1.set_ylabel("correlation E")
        ax1.set_title(f"Correlation vs Bob's angle, dots = {shots} shots", fontsize=10)
        ax1.legend(loc="lower left", fontsize=8); ax1.grid(alpha=0.3)

        ax2.axvspan(-CLASSICAL_BOUND, CLASSICAL_BOUND, color="tab:green", alpha=0.15)
        for x in (-QUANTUM_BOUND, QUANTUM_BOUND):
            ax2.axvline(x, color="tab:red", ls="--", lw=1)
        for x in (-CLASSICAL_BOUND, CLASSICAL_BOUND):
            ax2.axvline(x, color="tab:green", lw=1)
        ax2.plot([S_exact], [0.35], "k|", markersize=22, label=f"exact S = {S_exact:+.3f}")
        ax2.plot([S_sampled], [0.35], "o", color="tab:blue", markersize=11, label=f"sampled S = {S_sampled:+.3f}")
        ax2.text(0, 0.78, "local hidden variables: |S| <= 2", ha="center", fontsize=9, color="tab:green")
        ax2.text(QUANTUM_BOUND, 0.6, "2.83", ha="center", fontsize=8, color="tab:red")
        ax2.text(-QUANTUM_BOUND, 0.6, "-2.83", ha="center", fontsize=8, color="tab:red")
        ax2.set_xlim(-3.2, 3.2); ax2.set_ylim(0, 1); ax2.set_yticks([])
        ax2.set_xlabel("CHSH value S"); ax2.set_title("Bell test", fontsize=10)
        ax2.legend(loc="lower center", fontsize=8)
        fig.tight_layout()
        return fig, vals, S_exact, S_sampled

    def table_html(self, vals, S_exact, S_sampled):
        rows = "".join(f"<tr><td>{k}</td><td>{ex:+.3f}</td><td>{sa:+.3f}</td></tr>" for k, (ex, sa) in vals.items())
        verdict = ("<b>Violates the classical bound.</b> No assignment of pre-existing values to the qubits can produce this."
                   if abs(S_sampled) > CLASSICAL_BOUND else
                   "Within the classical bound. A product state, or badly chosen angles, never beats 2.")
        return (f"<table style='font-family:monospace; font-size:12px; border-collapse:collapse'>"
                f"<tr><th style='text-align:left'>pair</th><th>exact</th><th>sampled</th></tr>{rows}"
                f"<tr><td><b>S = E(a,b) - E(a,b') + E(a',b) + E(a',b')</b></td><td><b>{S_exact:+.3f}</b></td><td><b>{S_sampled:+.3f}</b></td></tr>"
                f"</table><div style='font-size:12px; margin-top:4px'>{verdict}</div>")

    # ---- widgets -----------------------------------------------------------
    def _build_ui(self, shots):
        self.state_dd = w.Dropdown(options=list(STATES), value=self.state_name, description="state", layout=w.Layout(width="290px"))
        self.shots = w.IntSlider(value=shots, min=100, max=4000, step=100, description="shots", continuous_update=False, layout=w.Layout(width="280px"))
        sl = lambda v, d: w.IntSlider(value=v, min=0, max=180, step=5, description=d, continuous_update=False, layout=w.Layout(width="280px"))
        self.a, self.a2, self.b, self.b2 = sl(0, "Alice a"), sl(90, "Alice a'"), sl(45, "Bob b"), sl(135, "Bob b'")
        preset_btns = []
        for name, angles in PRESETS.items():
            btn = w.Button(description=name, layout=w.Layout(width="230px"))
            btn.on_click(lambda _b, ang=angles: self._set_angles(ang))
            preset_btns.append(btn)
        self.out = w.Output()
        self.table = w.HTML()
        for widget in (self.state_dd, self.shots, self.a, self.a2, self.b, self.b2):
            widget.observe(lambda ch: self._render(), names="value")
        header = w.HTML("<b>Bell Test Lab.</b> Alice and Bob each choose two measurement angles in the X-Z plane. "
                        "Each correlation E is measured with the shot budget, then combined into the CHSH value S.")
        self.ui = w.VBox([header, w.HBox([self.state_dd, self.shots]), w.HBox([self.a, self.a2]), w.HBox([self.b, self.b2]),
                          w.HBox(preset_btns), self.out, self.table])

    def _set_angles(self, angles):
        # set silently, then render once
        for s, v in zip((self.a, self.a2, self.b, self.b2), angles):
            s.unobserve_all("value")
            s.value = v
            s.observe(lambda ch: self._render(), names="value")
        self._render()

    def _render(self):
        fig, vals, S_exact, S_sampled = self.figure()
        show_in(self.out, fig)
        self.table.value = self.table_html(vals, S_exact, S_sampled)


def bell_test_lab(state=None, shots=1000, static=None):
    """Open the Bell Test Lab. state: one of the keys in qclab.bell.STATES."""
    lab = BellTestLab(state, shots)
    if is_static(static):
        caption("Bell Test Lab (static preview). Run this cell in JupyterLab for the interactive version.")
        fig, vals, S_exact, S_sampled = lab.figure()
        display_static(fig)
        display(HTML(lab.table_html(vals, S_exact, S_sampled)))
        return None
    display(lab.ui)
    return None
