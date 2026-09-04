"""Grover Stepper: watch amplitude amplification one iteration at a time."""
import numpy as np
import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import display
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .common import show_in, is_static, caption, display_static


def phase_oracle(n, marked):
    qc = QuantumCircuit(n)
    zeros = [i for i, bit in enumerate(reversed(marked)) if bit == "0"]
    if zeros:
        qc.x(zeros)
    qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
    if zeros:
        qc.x(zeros)
    return qc


def diffuser(n):
    qc = QuantumCircuit(n)
    qc.h(range(n)); qc.x(range(n))
    qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
    qc.x(range(n)); qc.h(range(n))
    return qc


def grover_amplitudes(n, marked, k, half_step=False):
    # Real amplitudes after k full iterations (and optionally the oracle of iteration k+1), global sign removed.
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(k):
        qc.compose(phase_oracle(n, marked), inplace=True)
        qc.compose(diffuser(n), inplace=True)
    if half_step:
        qc.compose(phase_oracle(n, marked), inplace=True)
    amps = Statevector(qc).data.real
    return amps * (1 if amps.sum() >= 0 else -1)


def optimal_iterations(n):
    theta = np.arcsin(1 / np.sqrt(2 ** n))
    return int(np.floor(np.pi / (4 * theta))), theta


class GroverStepper:
    def __init__(self, n=3, marked="101"):
        self.n = int(n)
        self.marked = marked
        self._build_ui()
        self._render()

    def kmax(self):
        return 2 * optimal_iterations(self.n)[0] + 2

    def figure(self, k, show_reflections=True):
        n, marked = self.n, self.marked
        N = 2 ** n
        k_opt, theta = optimal_iterations(n)
        amps = grover_amplitudes(n, marked, k)
        labels = [format(i, f"0{n}b") for i in range(N)]
        idx = int(marked, 2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
        colors = ["tab:orange" if i == idx else ("tab:blue" if a >= 0 else "tab:red") for i, a in enumerate(amps)]
        ax1.bar(labels, amps, color=colors)
        ax1.axhline(amps.mean(), color="gray", ls="--", lw=1, label="mean amplitude")
        ax1.axhline(0, color="k", lw=0.8)
        ax1.set_ylim(-0.75, 1.05); ax1.set_ylabel("amplitude")
        ax1.set_title(f"after {k} iteration(s): P({marked}) = {amps[idx]**2:.3f}   (optimal k = {k_opt})", fontsize=10)
        if N > 8:
            ax1.tick_params(axis="x", rotation=90)
        ax1.legend(loc="upper left", fontsize=8)

        # geometric picture in the plane spanned by |w> (vertical) and |s_perp> (horizontal)
        t = np.linspace(0, 2 * np.pi, 200)
        ax2.plot(np.cos(t), np.sin(t), color="lightgray", lw=1)
        ax2.axhline(0, color="gray", lw=0.6); ax2.axvline(0, color="gray", lw=0.6)
        def arrow(angle, color, label, ls="-"):
            ax2.annotate("", xy=(np.cos(angle), np.sin(angle)), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color=color, lw=2, linestyle=ls))
            ax2.plot([], [], color=color, ls=ls, lw=2, label=label)
        arrow(theta, "gray", "start |s>")
        cur = (2 * k + 1) * theta
        arrow(cur, "tab:blue", f"now, angle {np.degrees(cur):.0f} deg")
        if show_reflections:
            arrow(-cur, "tab:red", "after oracle (flip about |s_perp>)", ls="--")
            arrow(cur + 2 * theta, "tab:green", "after diffuser (flip about |s>)", ls="--")
        ax2.set_xlim(-1.15, 1.15); ax2.set_ylim(-1.15, 1.15); ax2.set_aspect("equal")
        ax2.set_xlabel("|s_perp>  (all wrong answers)"); ax2.set_ylabel("|w>  (the answer)")
        ax2.set_title(f"each iteration rotates by 2 theta = {np.degrees(2*theta):.1f} deg", fontsize=10)
        ax2.legend(loc="lower left", fontsize=7)
        fig.tight_layout()
        return fig

    # ---- widgets -----------------------------------------------------------
    def _build_ui(self):
        self.n_dd = w.Dropdown(options=[2, 3, 4], value=self.n, description="qubits", layout=w.Layout(width="140px"))
        self.marked_txt = w.Text(value=self.marked, description="marked", layout=w.Layout(width="180px"))
        self.k = w.IntSlider(value=0, min=0, max=self.kmax(), step=1, description="iteration k", continuous_update=False, layout=w.Layout(width="320px"))
        self.play = w.Play(value=0, min=0, max=self.kmax(), step=1, interval=900, description="play")
        w.jslink((self.play, "value"), (self.k, "value"))
        self.reflect = w.Checkbox(value=True, description="show the two reflections", indent=False)
        prev = w.Button(description="Prev", layout=w.Layout(width="70px"))
        nxt = w.Button(description="Next", button_style="primary", layout=w.Layout(width="70px"))
        reset = w.Button(description="Reset", button_style="warning", layout=w.Layout(width="70px"))
        self.status = w.HTML()
        self.out = w.Output()

        self.n_dd.observe(lambda ch: self._on_n(ch["new"]), names="value")
        self.marked_txt.observe(lambda ch: self._on_marked(ch["new"]), names="value")
        self.k.observe(lambda ch: self._render(), names="value")
        self.reflect.observe(lambda ch: self._render(), names="value")
        prev.on_click(lambda _b: self._step(-1))
        nxt.on_click(lambda _b: self._step(+1))
        reset.on_click(lambda _b: setattr(self.k, "value", 0))

        header = w.HTML("<b>Grover Stepper.</b> Step through iterations. Left: the amplitude of every basis state, the answer in orange. "
                        "Right: the same state as a single arrow rotating towards the answer.")
        self.ui = w.VBox([header, w.HBox([self.n_dd, self.marked_txt, self.reflect, self.status]),
                          w.HBox([prev, nxt, reset, self.k, self.play]), self.out])

    def _on_n(self, n):
        self.n = int(n)
        self.marked = "1" * self.n if len(self.marked) != self.n else self.marked
        self.marked_txt.value = self.marked
        self.k.max = self.play.max = self.kmax()
        self.k.value = 0
        self._render()

    def _on_marked(self, text):
        text = text.strip()
        if len(text) != self.n or any(c not in "01" for c in text):
            self.status.value = f"<span style='color:#c00'>marked must be {self.n} characters of 0/1</span>"
            return
        self.status.value = ""
        self.marked = text
        self._render()

    def _step(self, d):
        self.k.value = int(np.clip(self.k.value + d, 0, self.k.max))

    def _render(self):
        show_in(self.out, self.figure(self.k.value, self.reflect.value))


def grover_stepper(n=3, marked="101", static=None):
    """Open the Grover Stepper for n qubits (2 to 4) and a marked bit-string."""
    lab = GroverStepper(n, marked)
    if is_static(static):
        caption("Grover Stepper (static preview at k = 1). Run this cell in JupyterLab for the interactive version.")
        display_static(lab.figure(1))
        return None
    display(lab.ui)
    return None
