"""Bloch Sphere Lab: press gates, watch the state move."""
import numpy as np
import ipywidgets as w
from IPython.display import display, HTML
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, state_fidelity
from .common import (bloch_coords, angles, rotation_path, bloch_figure, fmt_c, show_in,
                     is_static, caption, display_static, state_from_angles)

FIXED_GATES = [("X", "x"), ("Y", "y"), ("Z", "z"), ("H", "h"),
               ("S", "s"), ("S-dag", "sdg"), ("T", "t"), ("T-dag", "tdg")]
ROTATIONS = [("RX", "rx"), ("RY", "ry"), ("RZ", "rz")]


def _to_state(spec):
    if isinstance(spec, Statevector):
        return spec
    if isinstance(spec, str):
        return Statevector.from_label(spec)
    theta, phi = spec
    return state_from_angles(theta, phi)


class BlochLab:
    def __init__(self, start="0", target=None):
        self.start = _to_state(start)
        self.target = _to_state(target) if target is not None else None
        self.reset_state()
        self._build_ui()
        self._render()

    # ---- state -------------------------------------------------------------
    def reset_state(self):
        self.qc = QuantumCircuit(1)
        self.states = [self.start]
        self.paths = []
        self.names = []

    @property
    def sv(self):
        return self.states[-1]

    def apply(self, method, label, angle=None):
        prev = self.sv
        if angle is None:
            getattr(self.qc, method)(0)
            self.names.append(label)
        else:
            getattr(self.qc, method)(angle, 0)
            self.names.append(f"{label}({np.degrees(angle):.0f})")
        U = self.qc.data[-1].operation.to_matrix()
        self.paths.append(rotation_path(U, prev))
        self.states.append(prev.evolve(Operator(U)))

    def undo(self):
        if not self.names:
            return
        del self.qc.data[-1]
        self.states.pop()
        self.paths.pop()
        self.names.pop()

    # ---- views -------------------------------------------------------------
    def figure(self, size=4.2):
        vectors = [bloch_coords(self.sv)]
        colors = ["tab:blue"]
        if self.target is not None:
            vectors.append(bloch_coords(self.target))
            colors.append("tab:green")
        path = np.concatenate(self.paths) if self.paths else None
        title = " ".join(self.names) if self.names else "start"
        return bloch_figure(vectors, colors, path=path, title=title, size=size)

    def info_html(self):
        a, b = self.sv.data
        th, ph = angles(self.sv)
        U = Operator(self.qc).data
        rows = [
            f"<b>Sequence:</b> {', '.join(self.names) if self.names else '(nothing yet)'}",
            f"<b>State:</b> alpha = {fmt_c(a)} &nbsp; beta = {fmt_c(b)}",
            f"<b>Probabilities:</b> P(0) = {abs(a)**2:.3f} &nbsp; P(1) = {abs(b)**2:.3f}",
            f"<b>Bloch angles:</b> theta = {np.degrees(th):.1f} deg &nbsp; phi = {np.degrees(ph):.1f} deg",
            "<b>Matrix of the whole sequence:</b><br>"
            f"<code>[{fmt_c(U[0,0])} &nbsp; {fmt_c(U[0,1])}]<br>[{fmt_c(U[1,0])} &nbsp; {fmt_c(U[1,1])}]</code>",
        ]
        if self.target is not None:
            F = float(state_fidelity(self.sv, self.target))
            rows.append(f"<b>Fidelity to target:</b> {F:.3f}" + (" &nbsp; target reached" if F > 0.999 else ""))
        return ("<div style='font-family:monospace; font-size:12px; line-height:1.7'>"
                + "<br>".join(rows) + "</div>")

    # ---- widgets -----------------------------------------------------------
    def _build_ui(self):
        self.bloch_out = w.Output()
        self.circ_out = w.Output()
        self.info = w.HTML()

        gate_btns = []
        for label, method in FIXED_GATES:
            btn = w.Button(description=label, layout=w.Layout(width="68px"))
            btn.on_click(lambda _b, m=method, l=label: self._on_gate(m, l))
            gate_btns.append(btn)

        self.angle = w.IntSlider(value=90, min=-180, max=180, step=5, description="angle (deg)",
                                 continuous_update=False, layout=w.Layout(width="330px"))
        rot_btns = []
        for label, method in ROTATIONS:
            btn = w.Button(description=label, layout=w.Layout(width="68px"))
            btn.on_click(lambda _b, m=method, l=label: self._on_gate(m, l, use_angle=True))
            rot_btns.append(btn)

        undo = w.Button(description="Undo", button_style="warning", layout=w.Layout(width="90px"))
        undo.on_click(lambda _b: self._on_undo())
        reset = w.Button(description="Reset", button_style="danger", layout=w.Layout(width="90px"))
        reset.on_click(lambda _b: self._on_reset())

        legend = "blue = current state" + (", green = target" if self.target is not None else "") + ", orange = path taken"
        header = w.HTML(f"<b>Bloch Sphere Lab.</b> Press a gate and watch the arrow move ({legend}).")
        self.ui = w.VBox([
            header,
            w.HBox(gate_btns),
            w.HBox([self.angle] + rot_btns),
            w.HBox([undo, reset]),
            w.HBox([self.bloch_out, w.VBox([self.info, self.circ_out])]),
        ])

    def _on_gate(self, method, label, use_angle=False):
        self.apply(method, label, np.radians(self.angle.value) if use_angle else None)
        self._render()

    def _on_undo(self):
        self.undo()
        self._render()

    def _on_reset(self):
        self.reset_state()
        self._render()

    def _render(self):
        show_in(self.bloch_out, self.figure())
        self.info.value = self.info_html()
        show_in(self.circ_out, self.qc.draw("mpl", scale=0.7) if self.names else None)


def bloch_lab(start="0", target=None, static=None):
    """Open the Bloch Sphere Lab.

    start:  label ('0', '1', '+', '-', 'r', 'l') or (theta, phi) for the initial state.
    target: optional label or (theta, phi); the lab then reports the fidelity to it.
    """
    lab = BlochLab(start, target)
    if is_static(static):
        caption("Bloch Sphere Lab (static preview). Run this cell in JupyterLab for the interactive version.")
        display_static(lab.figure())
        display(HTML(lab.info_html()))
        return None
    display(lab.ui)
    return None
