# exp2d
Python/C code for simulation of exciton-polaritons in artificial 2D lattices. 

**Jupyter notebook interactive examples**

``linear-farfield.ipynb``: Calculation of the far-field radiation from a polariton lattice. Example for the Lieb lattice 
is below (Fig.S9a of the Supplementary Material of Ref.[1]):

![Alt text](/examples/lieb-far-field.png?raw=true "Lieb lattice in the far field (Fig.S9a of the Supplementary Material of Ref.[1])")

``linear-tetm.ipynb``: Linear spectrum in presence of a TE-TM splitting. For, example, 15th mode for the Lieb lattice at (kx,ky)=(0,0) is presented below.

![Alt text](/examples/lieb-tetm-nv15.png?raw=true "Probability density of a Bloch state in a Lieb lattice in presence of the TE-TM splitting")

``nonlinear-pump.ipynb``: Nonlinear evolution in presence of the TE-TM splitting and a coherent pumping.

![Alt text](/examples/square.gif?raw=true "Square lattice under coherent pumping")

**Supplementary files**

``README.md``: this file.

``exp2d.py``: Python module containing function definitions etc.

``exp2d.c``: C library (used by ``linear-farfield.ipynb`` and ``nonlinear-pump.ipynb``).

``Makefile``: Makefile to compile the supplementary C library (libexp2d.so produced as an output).

``examples``: A folder containing animation examples.

**Installation**

1. Clone:
    ```
    $ git clone https://github.com/drgulevich/exp2d.git
    ```
2. Make:
    ```
    $ make
    ```
3. Use:
    ```
    $ jupyter notebook linear-farfield.ipynb
    $ jupyter notebook linear-tetm.ipynb
    $ jupyter notebook nonlinear-pump.ipynb
    ```
The provided ``makefile`` is suitable for compilation on Linux machines. Modify the file accordingly to suit your needs. On MaC systems the flag
``-soname`` for compilation of the C shared library may need to be replaced by ``-install_name``.

**Research papers using exp2d**

1. C. E. Whittaker, E. Cancellieri, P. M. Walker, D. R. Gulevich, H. Schomerus, D. Vaitiekus, B. Royall, D. M. Whittaker, E. Clarke, I. V. Iorsh, I. A. Shelykh, M. S. Skolnick, and D. N. Krizhanovskii, "Exciton Polaritons in a Two-Dimensional Lieb Lattice with Spin-Orbit Coupling", Phys. Rev. Lett. 120, 097401 (2018). https://arxiv.org/abs/1705.03006.

2. D. R. Gulevich and D. Yudin, "Mimicking graphene with polaritonic spin vortices", Phys. Rev. B 96, 115433 (2017). https://arxiv.org/abs/1707.09195.
