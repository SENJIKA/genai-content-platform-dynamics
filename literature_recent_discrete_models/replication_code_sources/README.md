# Provenance of public code and data screened for the numerical extension

The local research workspace preserves the exact public inputs screened for the
paper.  The public GitHub repository intentionally excludes third-party binary
inputs and extracted code, while retaining this provenance record and download
instructions.  These materials are references, not a code base copied into the
paper's solver.

## Local-only files and verified sources

- `DataEconomyReplication_June2026_verified.zip`: Farboodi and Veldkamp,
  *A Model of the Data Economy*, official 2026 Zenodo replication package,
  DOI `10.5281/zenodo.18378461`.  Verified MD5:
  `d6ce7c32a73bf6dbfec599958ddd2db6`.
- `data_economy_verified/`: clean extraction of that verified archive.  The
  numerical workflow was inspected for its use of state grids, interpolation,
  value iteration, and forward simulation.  No Julia code or calibrated
  parameter vector was copied.
- `StructuralRL_SRL_main.zip` and `structural_rl/SRL-main/`: the public MIT
  licensed tutorial repository accompanying Yang, Wang, Schaab, and Moll
  (2025).  Its generic VFI baseline was inspected for convergence and residual
  checks.  The solver in `scripts/solve_dynamic_equilibria.py` was written
  independently in NumPy.
- `kuairec_caption_category.csv`: the category/caption table from the KuaiRec
  open dataset, DOI `10.5281/zenodo.18164998`.  Verified MD5:
  `31bc38cdccdf75a71df137779035f8cb`.  Only the first-level category frequency
  distribution is used to form normalized Shannon diversity; no engagement
  outcome is used as an AI-short-drama payoff parameter.

## Deliberate boundaries

The resulting exercise is a partial calibration and mechanism check, not a
structural estimate.  Discounting, depreciation, cost curvature, and payoff
scales are transparent normalizations subjected to sensitivity checks.  A
public short-video category moment is not treated as evidence about AI short
drama contract levels.  Negative implementation payments are kept separate
from non-price quotas, so a quota never creates fictitious transfer revenue.
