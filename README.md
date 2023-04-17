# VMAP4Fenics
VMAP4Fenics is a wrapper that acts as a bridge between the Python interface of VMAP and Finite Element Method (FEM) simulations conducted using Fenics. The Wrapper extracts the geometry describing a Fenics problem from a Dolfin Vector Function Space and stores the geometry among other information in an HDF5 file. This Project is integrated in FenicsConrete and LeBeDigital storing the results of concrete simulations conducted by [BAM](https://www.bam.de/Navigation/EN/Home/home.html) (Bundesanstalt für Materialforschung und -prüfung) the German Federal Institute for Materials Research and Testing.
## Installation
Requieres [Mamba installation](https://mamba.readthedocs.io/en/latest/installation.html)
> mamba install -c bam77 vmap4fenics
## Related Projects
* [VMAP](https://www.scai.fraunhofer.de/en/projects/VMAP.html)
* [FEniCS](https://fenicsproject.org//)
* [FenicsConcrete](https://github.com/BAMresearch/FenicsConcrete/)
* [LeBeDigital](https://github.com/BAMresearch/LebeDigital)