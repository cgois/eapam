# Code and results for ([arXiv:2205.05171](https://arxiv.org/abs/2205.05171))

- `detpoints.py` implements the listing of deterministic strategies for prepare and measure scenarios, directly the formats of PANDA or lrs for later facet enumeration.
- `eacc.py` implements a see-saw algorithm (Sec. IV of https://doi.org/10.1103/PRXQuantum.2.040357) to find lower-bounds on the success probability of the ambiguous guessing game described in the paper.
- `2-3-4-1.ext` and `2-3-4-1.ine` are the deterministic strategies and polytope facets for the (d,B,X,Y) = (2,3,4,1) prepare and measure scenario.
- `optmeas.npy` and `optstates.npy` are the optimal measurements and states achieving the bound of $4.174$ in the game described in Sec. III.B of the related paper.
