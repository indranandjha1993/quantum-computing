"""Guess the State: a hidden qubit, a shot budget, three measurement bases, one guess."""
import numpy as np
import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import display
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from .common import bloch_coords, bloch_figure, state_from_angles, show_in, is_static, caption

LABELS = {"0": "|0>", "1": "|1>", "+": "|+>", "-": "|->", "r": "|i>", "l": "|-i>"}
# gates that rotate the chosen axis onto Z before a standard measurement
BASIS_CHANGE = {"Z": [], "X": ["h"], "Y": ["sdg", "h"]}
AXIS = {"Z": "z", "X": "x", "Y": "y"}


class GuessGame:
    def __init__(self, difficulty="easy", budget=300, seed=None):
        if difficulty not in ("easy", "hard"):
            raise ValueError("difficulty must be 'easy' or 'hard'")
        self.difficulty = difficulty
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)
        self.history = []
        self._build_ui()
        self.new_round()

    # ---- game logic --------------------------------------------------------
    def new_round(self):
        if self.difficulty == "easy":
            self.hidden_label = str(self.rng.choice(list(LABELS)))
            self.hidden = Statevector.from_label(self.hidden_label)
        else:
            self.hidden_label = None
            theta = float(np.arccos(self.rng.uniform(-1, 1)))
            phi = float(self.rng.uniform(0, 2 * np.pi))
            self.hidden = state_from_angles(theta, phi)
        self.counts = {b: {"0": 0, "1": 0} for b in "ZXY"}
        self.left = self.budget
        self.revealed = False
        self.hint_used = False
        self.status.value = ""
        self.hint.value = ""
        with self.reveal_out:
            self.reveal_out.clear_output()
        self._render_results()

    def measure(self, basis, shots):
        if self.revealed:
            return "This round is over. Press New round."
        if shots > self.left:
            return f"Only {self.left} shots left in the budget."
        qc = QuantumCircuit(1)
        for g in BASIS_CHANGE[basis]:
            getattr(qc, g)(0)
        sample = self.hidden.evolve(qc).sample_counts(shots)
        for k in ("0", "1"):
            self.counts[basis][k] += int(sample.get(k, 0))
        self.left -= shots
        return ""

    def estimates(self):
        est = {}
        for b in "ZXY":
            n0, n1 = self.counts[b]["0"], self.counts[b]["1"]
            N = n0 + n1
            est[AXIS[b]] = (n0 - n1) / N if N else None
        return est

    def submit(self, guess):
        F = float(state_fidelity(self.hidden, guess))
        self.revealed = True
        self.history.append(F)
        return F

    # ---- views -------------------------------------------------------------
    def results_figure(self):
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
        for ax, b in zip(axes, "ZXY"):
            c = self.counts[b]
            ax.bar(["0", "1"], [c["0"], c["1"]], color=["tab:blue", "tab:red"])
            ax.set_title(f"{b} basis, {c['0'] + c['1']} shots", fontsize=10)
            ax.set_ylim(0, max(10, c["0"] + c["1"]))
        fig.suptitle(f"Measurements so far. Budget left: {self.left} shots", fontsize=11)
        fig.tight_layout()
        return fig

    def reveal_figure(self, guess):
        return bloch_figure([bloch_coords(self.hidden), bloch_coords(guess)],
                            ["tab:green", "tab:blue"], title="green = hidden, blue = your guess", size=3.8)

    def hint_html(self):
        est = self.estimates()
        parts = []
        for axis in "xyz":
            v = est[axis]
            parts.append(f"{axis} = {v:+.2f}" if v is not None else f"{axis} = ? (not measured)")
        return ("<span style='font-family:monospace'>Estimated Bloch coordinates from your counts: "
                + ", &nbsp;".join(parts) + "</span><br><small>P(0) in a basis is (1 + coordinate)/2, "
                "so coordinate = (n0 - n1) / shots. That is state tomography.</small>")

    # ---- widgets -----------------------------------------------------------
    def _build_ui(self):
        self.basis = w.Dropdown(options=["Z", "X", "Y"], value="Z", description="basis", layout=w.Layout(width="130px"))
        self.shots = w.Dropdown(options=[10, 25, 50, 100], value=25, description="shots", layout=w.Layout(width="140px"))
        measure = w.Button(description="Measure", button_style="primary", layout=w.Layout(width="100px"))
        hint_btn = w.Button(description="Hint", layout=w.Layout(width="80px"))
        newround = w.Button(description="New round", button_style="success", layout=w.Layout(width="110px"))
        submit = w.Button(description="Submit guess", button_style="info", layout=w.Layout(width="120px"))

        if self.difficulty == "easy":
            self.guess_label = w.Dropdown(options=[(v, k) for k, v in LABELS.items()], description="guess",
                                          layout=w.Layout(width="160px"))
            guess_row = [self.guess_label, submit]
        else:
            self.guess_theta = w.IntSlider(value=90, min=0, max=180, step=5, description="theta (deg)",
                                           continuous_update=False, layout=w.Layout(width="300px"))
            self.guess_phi = w.IntSlider(value=0, min=0, max=355, step=5, description="phi (deg)",
                                         continuous_update=False, layout=w.Layout(width="300px"))
            guess_row = [self.guess_theta, self.guess_phi, submit]

        self.status = w.HTML()
        self.hint = w.HTML()
        self.score = w.HTML()
        self.results_out = w.Output()
        self.reveal_out = w.Output()

        measure.on_click(lambda _b: self._on_measure())
        hint_btn.on_click(lambda _b: self._on_hint())
        submit.on_click(lambda _b: self._on_submit())
        newround.on_click(lambda _b: self.new_round())

        rules = ("<b>Guess the State.</b> A hidden qubit has been prepared. Spend your shot budget measuring it in "
                 "the Z, X or Y basis, then guess the state. "
                 + ("Easy mode: it is one of |0>, |1>, |+>, |->, |i>, |-i>." if self.difficulty == "easy"
                    else "Hard mode: any point on the Bloch sphere, guess theta and phi."))
        self.ui = w.VBox([
            w.HTML(rules),
            w.HBox([self.basis, self.shots, measure, hint_btn, newround, self.status]),
            self.results_out,
            self.hint,
            w.HBox(guess_row),
            self.reveal_out,
            self.score,
        ])

    def _current_guess(self):
        if self.difficulty == "easy":
            return Statevector.from_label(self.guess_label.value), LABELS[self.guess_label.value]
        th, ph = np.radians(self.guess_theta.value), np.radians(self.guess_phi.value)
        return state_from_angles(th, ph), f"theta = {self.guess_theta.value} deg, phi = {self.guess_phi.value} deg"

    def _on_measure(self):
        msg = self.measure(self.basis.value, self.shots.value)
        self.status.value = f"<span style='color:#c00'>{msg}</span>" if msg else ""
        self._render_results()

    def _on_hint(self):
        self.hint_used = True
        self.hint.value = self.hint_html()

    def _on_submit(self):
        if self.revealed:
            self.status.value = "<span style='color:#c00'>Already revealed. Press New round.</span>"
            return
        guess, desc = self._current_guess()
        F = self.submit(guess)
        show_in(self.reveal_out, self.reveal_figure(guess))
        if F > 0.98:
            verdict = "Excellent, that is the state."
        elif F > 0.9:
            verdict = "Close. A few more shots in the weakest basis would nail it."
        elif F > 0.7:
            verdict = "Getting there. Did you measure in all three bases?"
        else:
            verdict = "Way off. Remember: Z tells you z, X tells you x, Y tells you y."
        hidden = LABELS[self.hidden_label] if self.hidden_label else "the hidden state"
        used = self.budget - self.left
        avg = float(np.mean(self.history))
        self.score.value = (f"<b>Your guess:</b> {desc} &nbsp; <b>Fidelity with {hidden}:</b> {F:.3f} &nbsp; "
                            f"<b>Shots used:</b> {used}" + (" (hint used)" if self.hint_used else "")
                            + f"<br>{verdict}<br><small>Rounds played: {len(self.history)}, average fidelity {avg:.3f}</small>")

    def _render_results(self):
        show_in(self.results_out, self.results_figure())


def guess_the_state(difficulty="easy", budget=300, seed=None, static=None):
    """Open the Guess the State game ('easy' or 'hard')."""
    game = GuessGame(difficulty, budget, seed)
    if is_static(static):
        caption(f"Guess the State, {difficulty} mode (static preview). Run this cell in JupyterLab to play.")
        caption(f"Rules: a hidden qubit is prepared. You have {budget} shots to measure it in the Z, X and Y bases, "
                "then you guess the state and see the fidelity of your guess.")
        return None
    display(game.ui)
    return None
