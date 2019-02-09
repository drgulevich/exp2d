# exp2d
Python/C code for simulation of exciton-polaritons in artificial 2D lattices.

**Jupyter notebook interactive examples**

``linear-farfield.ipynb``: calculation of the far-field radiation from the Lieb lattice, see Ref.[1] for details.

![Alt text](/examples/lieb-far-field.png?raw=true "Lieb lattice in the far field (Fig.S9a of the Supplementary Material of Ref.[1])")

``linear-tetm.ipynb``: linear spectrum in presence of a TE-TM splitting.

![Alt text](/examples/lieb-tetm-nv15.png?raw=true "Probability density of a Bloch state in a Lieb lattice in presence of the TE-TM splitting")

``nonlinear-pump.ipynb``: time-evolution in the nonlinear regime is presence of a TE-TM splitting and coherent pumping.

![Alt text](/examples/square.gif?raw=true "Square lattice under coherent pumping")

**Supplementary files**

``README.md``: this file.

``exp2d.c``: C library (used by ``linear-farfield.ipynb`` and ``nonlinear-pump.ipynb``).

``Makefile``: Makefile to compile the supplementary C library (libexp2d.so produced as an output).

``examples``: A folder containing animation examples.

**Research papers using exp2d**

1. C. E. Whittaker, E. Cancellieri, P. M. Walker, D. R. Gulevich, H. Schomerus, D. Vaitiekus, B. Royall, D. M. Whittaker, E. Clarke, I. V. Iorsh, I. A. Shelykh, M. S. Skolnick, and D. N. Krizhanovskii, "Exciton Polaritons in a Two-Dimensional Lieb Lattice with Spin-Orbit Coupling", Phys. Rev. Lett. 120, 097401 (2018).
