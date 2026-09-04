"""Interactive labs for the quantum computing course.

    from qclab import bloch_lab, circuit_composer, guess_the_state, quiz
    from qclab import bell_test_lab, grover_stepper, teleport_walkthrough, noise_dial
"""
from .bloch import bloch_lab, BlochLab
from .composer import circuit_composer, CircuitComposer
from .game import guess_the_state, GuessGame
from .quiz import quiz, QUESTIONS
from .bell import bell_test_lab, BellTestLab
from .grover import grover_stepper, GroverStepper
from .teleport import teleport_walkthrough, TeleportWalkthrough
from .noise import noise_dial, NoiseDial

__all__ = ["bloch_lab", "BlochLab", "circuit_composer", "CircuitComposer",
           "guess_the_state", "GuessGame", "quiz", "QUESTIONS",
           "bell_test_lab", "BellTestLab", "grover_stepper", "GroverStepper",
           "teleport_walkthrough", "TeleportWalkthrough", "noise_dial", "NoiseDial"]
