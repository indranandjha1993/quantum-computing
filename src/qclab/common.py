"""Shared helpers for the qclab interactive labs."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from IPython.display import display, clear_output
from qiskit.quantum_info import Statevector
from qiskit.visualization.bloch import Bloch

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def is_static(flag=None):
    # Static mode renders a plain figure instead of widgets. Used when notebooks are
    # executed headlessly so the saved outputs still show something useful.
    if flag is not None:
        return bool(flag)
    return os.environ.get("QCLAB_STATIC", "") == "1"


def caption(text):
    print(text)


def display_static(*figs):
    for f in figs:
        if f is not None:
            display(f)
            plt.close(f)


def show_in(out, *figs):
    # Replace the contents of an Output widget with the given figures.
    with out:
        clear_output(wait=True)
        for f in figs:
            if f is not None:
                display(f)
                plt.close(f)


def fmt_c(z, digits=3):
    z = complex(z)
    re, im = round(z.real, digits), round(z.imag, digits)
    if abs(im) < 10 ** -digits:
        return f"{re:+.{digits}f}"
    if abs(re) < 10 ** -digits:
        return f"{im:+.{digits}f}i"
    return f"{re:+.{digits}f}{im:+.{digits}f}i"


def bloch_coords(sv):
    a, b = np.asarray(sv.data, dtype=complex)[:2]
    ab = np.conj(a) * b
    return float(2 * ab.real), float(2 * ab.imag), float(abs(a) ** 2 - abs(b) ** 2)


def angles(sv):
    x, y, z = bloch_coords(sv)
    theta = float(np.arccos(np.clip(z, -1, 1)))
    phi = float(np.arctan2(y, x) % (2 * np.pi))
    return theta, phi


def state_from_angles(theta, phi):
    return Statevector([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def axis_angle(U):
    # Decompose a 2x2 unitary as a rotation of the Bloch sphere: returns (axis, angle).
    U = np.asarray(U, dtype=complex)
    U = U / np.sqrt(np.linalg.det(U))
    if U[0, 0].real < 0:
        U = -U
    c = float(np.clip(U[0, 0].real, -1, 1))
    s = float(np.sqrt(max(0.0, 1 - c * c)))
    gamma = 2 * np.arccos(c)
    if s < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    nz = -U[0, 0].imag / s
    nx = -U[0, 1].imag / s
    ny = -U[0, 1].real / s
    return np.array([nx, ny, nz]), float(gamma)


def rotation_path(U, sv, steps=24):
    # Points on the Bloch sphere along the rotation that takes sv to U sv.
    n, gamma = axis_angle(U)
    nsig = n[0] * SX + n[1] * SY + n[2] * SZ
    pts = []
    for t in np.linspace(0, 1, steps):
        Ut = np.cos(gamma * t / 2) * np.eye(2) - 1j * np.sin(gamma * t / 2) * nsig
        pts.append(bloch_coords(Statevector(Ut @ sv.data)))
    return np.array(pts)


def bloch_figure(vectors, colors, path=None, title="", size=4.2):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(projection="3d")
    b = Bloch(fig=fig, axes=ax)
    b.vector_color = list(colors)
    b.point_color = ["tab:orange"]
    b.point_marker = ["o"]
    b.point_size = [12]
    for v in vectors:
        b.add_vectors(list(v))
    if path is not None and len(path) > 1:
        b.add_points([path[:, 0], path[:, 1], path[:, 2]], meth="l")
    b.render(title=title)
    return fig


def phase_bars(sv, ax, title="amplitudes: height = |a|, colour = phase"):
    amps = np.asarray(sv.data)
    n = sv.num_qubits
    labels = [format(i, f"0{n}b") for i in range(len(amps))]
    mags = np.abs(amps)
    phases = np.angle(amps) % (2 * np.pi)
    colors = [hsv_to_rgb((p / (2 * np.pi), 0.85, 0.9)) if m > 1e-9 else (0.85, 0.85, 0.85)
              for p, m in zip(phases, mags)]
    ax.bar(labels, mags, color=colors)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("|amplitude|")
    ax.set_title(title, fontsize=10)
    for i, (m, p) in enumerate(zip(mags, phases)):
        if m > 1e-9:
            ax.text(i, m + 0.02, f"{np.degrees(p):.0f} deg", ha="center", fontsize=7)
    if len(labels) > 8:
        ax.tick_params(axis="x", rotation=90)
