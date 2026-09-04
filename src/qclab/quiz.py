"""Checkpoint quizzes, one bank per module."""
import html
import ipywidgets as w
from IPython.display import display
from .common import is_static, caption

# Each question: text, options, index of the correct option, short explanation.
QUESTIONS = {
    1: [
        ("A qubit has amplitudes alpha = 0.6 and beta = 0.8. What is the probability of measuring 1?",
         ["0.8", "0.64", "0.36", "0.5"], 1,
         "Probability is the squared magnitude of the amplitude: 0.8 squared is 0.64."),
        ("Which of these is NOT a valid qubit state?",
         ["(1/sqrt2, 1/sqrt2)", "(1, 0)", "(0.5, 0.5)", "(0.6, 0.8i)"], 2,
         "0.25 + 0.25 = 0.5, so (0.5, 0.5) is not normalised. The others sum to 1."),
        ("Two states on the equator of the Bloch sphere differ only in phi. Measured in the Z basis they give",
         ["identical 50/50 statistics", "different probabilities", "always 0", "always 1"], 0,
         "Z measurement only sees the latitude. Phase is invisible until you change basis (Module 4)."),
        ("You measure a qubit in superposition and get 1. You measure it again immediately. You get",
         ["1 with certainty", "0 or 1 at random", "0 with certainty", "an error"], 0,
         "Measurement collapses the state to |1>. Measuring |1> again always gives 1."),
    ],
    2: [
        ("What does the Z gate do to |+>?",
         ["nothing", "turns it into |->", "turns it into |1>", "turns it into |0>"], 1,
         "Z flips the sign of the |1> amplitude, which moves |+> to the opposite side of the equator: |->."),
        ("Which of these gates is its own inverse?",
         ["T", "S", "H", "RZ(pi/2)"], 2,
         "H applied twice is the identity. T and S need their daggers to undo them."),
        ("RY(theta) applied to |0> gives P(1) equal to",
         ["sin^2(theta/2)", "cos^2(theta/2)", "sin(theta)", "theta / pi"], 0,
         "RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>, so P(1) = sin^2(theta/2)."),
        ("The sequence H, Z, H is the same as which single gate?",
         ["X", "Z", "I", "Y"], 0,
         "H swaps the X and Z axes, so conjugating Z by H gives X."),
    ],
    3: [
        ("In Qiskit the measured bitstring '01' means",
         ["qubit 0 = 0 and qubit 1 = 1", "qubit 0 = 1 and qubit 1 = 0", "both qubits are 1", "the circuit failed"], 1,
         "Qiskit is little-endian: the rightmost character is qubit 0."),
        ("Which two-qubit state is entangled?",
         ["(|00> + |01>)/sqrt2", "|01>", "(|00> + |01> + |10> + |11>)/2", "(|00> + |11>)/sqrt2"], 3,
         "The first is |0> on qubit 1 times |+> on qubit 0, the third is |+>|+>. Only the Bell state cannot be factored."),
        ("The purity of one qubit taken from a Bell pair is",
         ["1", "0.5", "0", "0.25"], 1,
         "Alone, each qubit of a Bell pair is maximally mixed, purity 1/2. A product-state qubit has purity 1."),
        ("Which circuit turns |00> into the Bell state (|00> + |11>)/sqrt2?",
         ["CX(0,1) then H on qubit 0", "H on both qubits", "X on qubit 1 then CX(0,1)", "H on qubit 0 then CX(0,1)"], 3,
         "H creates the superposition on the control, then CNOT copies the branch onto the target."),
    ],
    4: [
        ("How does the statistical error of a probability estimate shrink with the number of shots N?",
         ["like 1/N", "like 1/sqrt(N)", "it does not shrink", "like log N"], 1,
         "Standard error is sqrt(p(1-p)/N). Ten times the shots gives about 3.2 times the precision."),
        ("To tell |+> from |-> with a measurement you should",
         ["measure in Z many more times", "apply H, then measure", "apply X, then measure", "apply Z, then measure"], 1,
         "H maps the X axis onto the Z axis: H|+> = |0> and H|-> = |1>."),
        ("H, then RZ(pi), then H applied to |0> gives",
         ["|0> with certainty", "|1> with certainty", "50/50", "|+>"], 1,
         "P(0) = cos^2(phi/2) = cos^2(pi/2) = 0, so the result is always 1."),
        ("Destructive interference happens when",
         ["probabilities are negative", "amplitudes for the same outcome have opposite signs",
          "two qubits are entangled", "there are too few shots"], 1,
         "Amplitudes add before squaring, so +1/2 and -1/2 for the same outcome cancel to zero."),
    ],
    5: [
        ("How many classical bits must Alice send to Bob in teleportation?",
         ["0", "1", "2", "one per amplitude"], 2,
         "Her two measurement results select one of Bob's four possible corrections."),
        ("After the protocol, Alice's original qubit is",
         ["an unchanged copy", "entangled with Bob's qubit", "collapsed by her measurement", "in |+>"], 2,
         "She measured it. That is why teleportation does not violate no-cloning."),
        ("Why can teleportation not send information faster than light?",
         ["the qubits are too slow", "Bob's qubit is useless until the classical bits arrive",
          "entanglement breaks over distance", "Alice must wait for Bob's reply"], 1,
         "Without the two bits Bob holds one of four random twists of the state and cannot tell which."),
        ("Alice's bit from qubit 1 (the CNOT target) is 1. Bob applies",
         ["X", "Z", "H", "nothing"], 0,
         "The qubit-1 bit controls the X correction; the qubit-0 bit controls the Z correction."),
    ],
    6: [
        ("In Deutsch-Jozsa, measuring all zeros on the input register means the function is",
         ["balanced", "constant", "random", "undefined"], 1,
         "All signs agree, so the amplitudes interfere constructively into the all-zeros state."),
        ("Phase kickback needs the output qubit prepared in",
         ["|0>", "|1>", "|+>", "|->"], 3,
         "XOR-ing 1 into |-> flips its sign; the sign lands on the input branch."),
        ("For n = 4 inputs (16 values) the classical worst case needs how many evaluations?",
         ["4", "8", "9", "16"], 2,
         "Half the inputs plus one: 2^(n-1) + 1 = 9."),
        ("The Bernstein-Vazirani circuit returns",
         ["the secret string s directly", "constant or balanced", "one random bit of s", "the parity of s"], 0,
         "After the final Hadamards the register is exactly |s>."),
    ],
    7: [
        ("One Grover iteration consists of",
         ["measure, then re-prepare", "oracle sign flip, then reflection about the mean",
          "two Hadamards", "a CNOT and a Toffoli"], 1,
         "The oracle marks the answer with a minus sign; the diffuser turns that sign into a larger amplitude."),
        ("For N = 64 items, roughly how many Grover iterations are optimal?",
         ["32", "6", "64", "1"], 1,
         "About pi/4 times sqrt(64) = 6.3, so 6."),
        ("Running twice the optimal number of iterations",
         ["doubles the success probability", "changes nothing", "lowers the success probability", "guarantees success"], 2,
         "The state keeps rotating past the target. Success follows sin^2((2k+1) theta)."),
        ("Grover's speed-up over classical search is",
         ["exponential", "quadratic", "linear", "none"], 1,
         "sqrt(N) queries instead of N."),
    ],
    8: [
        ("Which error source dominates on current superconducting hardware?",
         ["single-qubit gates", "two-qubit gates", "idle qubits", "the classical computer"], 1,
         "Two-qubit gates are typically ten times worse than single-qubit gates."),
        ("Why does transpiling to a line-connected device add CNOT gates?",
         ["the device has extra qubits", "non-adjacent pairs need SWAPs, each SWAP is three CNOTs",
          "CNOT is the only gate the device knows", "to reduce readout error"], 1,
         "Routing moves qubits next to each other with SWAP gates."),
        ("With a physical bit-flip rate p = 0.1, the three-qubit repetition code has a logical error rate of about",
         ["0.1", "0.3", "0.028", "0.001"], 2,
         "3p^2 - 2p^3 = 0.03 - 0.002 = 0.028."),
        ("The three-qubit bit-flip code cannot correct",
         ["a single X error", "a Z (phase) error", "a flip on qubit 2", "readout of the majority"], 1,
         "It only detects flips in the Z basis. Phase errors pass through unnoticed."),
    ],
}


class Quiz:
    def __init__(self, module):
        if module not in QUESTIONS:
            raise ValueError(f"no quiz for module {module}")
        self.module = module
        self.items = QUESTIONS[module]
        self.correct = 0
        self.answered = 0
        self._build_ui()

    def _build_ui(self):
        blocks = []
        self.score = w.HTML()
        for i, (text, options, answer, why) in enumerate(self.items, start=1):
            q = w.HTML(f"<b>Q{i}.</b> {html.escape(text)}")
            radio = w.RadioButtons(options=options, value=None, layout=w.Layout(width="auto"))
            check = w.Button(description="Check", layout=w.Layout(width="80px"))
            feedback = w.HTML()
            check.on_click(lambda _b, r=radio, c=check, f=feedback, a=answer, why=why, opts=options: self._check(r, c, f, a, why, opts))
            blocks.append(w.VBox([q, radio, w.HBox([check, feedback])], layout=w.Layout(margin="0 0 12px 0")))
        header = w.HTML(f"<b>Checkpoint quiz, Module {self.module}.</b> {len(self.items)} questions. Pick an answer and press Check.")
        self.ui = w.VBox([header] + blocks + [self.score])
        self._update_score()

    def _check(self, radio, check, feedback, answer, why, options):
        if radio.value is None:
            feedback.value = "<span style='color:#c00'>Pick an answer first.</span>"
            return
        chosen = options.index(radio.value)
        self.answered += 1
        if chosen == answer:
            self.correct += 1
            feedback.value = f"<span style='color:#080'><b>Correct.</b> {html.escape(why)}</span>"
        else:
            feedback.value = (f"<span style='color:#c00'><b>Not quite.</b> The answer is \"{html.escape(options[answer])}\". "
                              f"{html.escape(why)}</span>")
        radio.disabled = True
        check.disabled = True
        self._update_score()

    def _update_score(self):
        n = len(self.items)
        if self.answered == n:
            note = " Well done, move on." if self.correct == n else " Re-read the sections for the ones you missed before moving on."
            self.score.value = f"<b>Score: {self.correct} / {n}.</b>{note}"
        else:
            self.score.value = f"<b>Score so far: {self.correct} / {self.answered}</b> (of {n})"


def quiz(module, static=None):
    """Show the checkpoint quiz for a module (1 to 8)."""
    q = Quiz(module)
    if is_static(static):
        caption(f"Checkpoint quiz, Module {module} (static preview). Run this cell in JupyterLab to answer interactively.")
        for i, (text, options, _a, _w) in enumerate(q.items, start=1):
            caption(f"\nQ{i}. {text}")
            for j, opt in enumerate(options):
                caption(f"    {'abcd'[j]}) {opt}")
        return None
    display(q.ui)
    return None
