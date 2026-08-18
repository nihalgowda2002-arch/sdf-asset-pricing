# The Stochastic Discount Factor and the CAPM

Asset pricing developed from a single equation, `p = E[mx]`, with the CAPM and the
consumption-based model derived as special cases. Includes the theory, a calibration to
standard post-war US moments, and a simulated cross-section demonstrating what a
single-factor model does when a priced factor is omitted.

Everything here is reproducible: seeds are fixed, and running the script regenerates every
number and figure in the write-up exactly.

## Headline results

| Result | Value |
|---|---|
| Risk aversion required to clear the Hansen–Jagannathan bound | γ = 25.8 |
| Risk aversion required at the observed consumption–return correlation (ρ = 0.20) | γ = 133.1 |
| Model-implied real risk-free rate at γ = 27.8 | 50.6% (observed ≈ 1%) |
| Monte Carlo check of the pricing equation, 2,000,000 draws | E[mR] = 0.999985 (s.e. 0.000138) |
| GRS test of the single-factor model on the simulated cross-section | J = 3.61, p = 1.1 × 10⁻⁴ |
| Same test after adding the omitted factor | max \|t(α)\| = 1.49 |

## Contents

| File | Description |
|---|---|
| `sdf_analysis.py` | All numerical work. Produces every table entry and figure. |
| `SDF-and-the-CAPM.pdf` | Full write-up, 21 pages: proofs, calibration, numerical results, code appendix. |
| `SDF-and-the-CAPM_slides.pdf` | Ten-slide summary of the argument. |

## Running it

```bash
pip install -r requirements.txt
python sdf_analysis.py
```

Runtime is a few seconds. Output goes to stdout; the three figures are written to the
working directory as `fig_hj.pdf`, `fig_puzzle.pdf` and `fig_sml.pdf`.

To capture the numerical output:

```bash
python sdf_analysis.py > results.txt
```

## What the code does

**Calibration and the puzzles.** Computes the Hansen–Jagannathan bound implied by the
observed Sharpe ratio, the coefficient of variation of the CRRA discount factor
`m = δ exp(−γΔc)` as γ varies, and the risk-free rate implied by the same parameter. The
two puzzles are the observation that no single γ satisfies both.

**Monte Carlo verification.** Simulates a lognormal endowment economy at γ = 10, δ = 0.98
with 2,000,000 draws (seed 4242) and checks that `E[mR] = 1` holds to within one standard
error, along with the closed-form risk-free rate and equity premium.

**Simulated cross-section.** Constructs an economy in which the true kernel has two priced
factors, forms ten test portfolios on an orthogonal 5 × 2 design of factor loadings, and
prices them with a single-factor model over 600 months (seed 20260817). Reports time-series
alphas with t-statistics, the GRS joint test, and cross-sectional regressions under both the
one- and two-factor specifications.

## Note on the data

The calibration inputs are documented post-war US real magnitudes of the order reported by
Mehra and Prescott (1985) and Campbell (2003):

| Quantity | Value |
|---|---|
| Mean log consumption growth | 2.00% |
| S.d. of log consumption growth | 1.35% |
| Mean equity excess return | 6.00% |
| S.d. of log equity return | 16.70% |
| Correlation of consumption growth with the equity return | 0.20 |
| Real risk-free rate | 1.00% |

**These are inputs, not estimates produced here.** No return or consumption series is
estimated in this repository. The conclusions depend on the order of magnitude of these
numbers, not on their third decimal.

The economy in the cross-sectional section is **synthetic**, generated from a two-factor
discount factor. Finding that a two-factor model prices it is therefore not evidence about
factor structure in real markets. The purpose is narrower: to verify the estimation
machinery on a case where the true answer is known, and to show that the observable
signature of an omitted priced factor is a systematic pattern of alphas aligned with
loadings on that factor.

## Main references

- Cochrane, *Asset Pricing*, revised ed., Princeton (2005) — the organising framework.
- Hansen and Jagannathan, *Journal of Political Economy* 99 (1991) — the volatility bound.
- Mehra and Prescott, *Journal of Monetary Economics* 15 (1985) — the equity premium puzzle.
- Weil, *Journal of Monetary Economics* 24 (1989) — the risk-free rate puzzle.
- Gibbons, Ross and Shanken, *Econometrica* 57 (1989) — the joint test of zero pricing errors.

Full reference list in the write-up.

## Author

Nihal Naresh Kumar — MSc Mathematical Finance

## Licence

MIT. See `LICENSE`.
