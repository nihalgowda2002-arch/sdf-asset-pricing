"""
Numerical work for: The Stochastic Discount Factor and the CAPM.
Every figure and every table entry quoted in the paper is produced here.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200})

# ---------------------------------------------------------------- calibration
g      = 0.0200   # mean log consumption growth
sig_c  = 0.0135   # s.d. of log consumption growth
mu_e   = 0.0600   # mean log equity excess return
sig_m  = 0.1670   # s.d. of log equity return
rho_cm = 0.2000   # corr(consumption growth, equity return)
rf_obs = 0.0100   # observed real risk-free rate
SR     = mu_e / sig_m
print("=== Table 5.1 calibration ===")
print(f"g={g} sig_c={sig_c} mu_e={mu_e} sig_m={sig_m} rho={rho_cm} rf={rf_obs}")
print(f"Sharpe ratio = {SR:.4f}")

# ---------------------------------------------------- Section 5: implied gamma
gam_hj_perfect = SR / sig_c                       # HJ bound, |rho|=1
gam_lognormal  = mu_e / (rho_cm * sig_c * sig_m)  # lognormal CCAPM, rho as observed
gam_lognormal_1 = mu_e / (1.0 * sig_c * sig_m)
print("\n=== Table 5.2 implied risk aversion ===")
print(f"HJ bound (|rho|=1)          gamma >= {gam_hj_perfect:.1f}")
print(f"lognormal CCAPM, rho=1      gamma  = {gam_lognormal_1:.1f}")
print(f"lognormal CCAPM, rho=0.20   gamma  = {gam_lognormal:.1f}")

def rf_model(gamma, delta=0.02):
    return delta + gamma * g - 0.5 * gamma ** 2 * sig_c ** 2

def prem_model(gamma, rho=rho_cm):
    return gamma * rho * sig_c * sig_m

print("\n=== Table 5.3 risk-free rate puzzle (delta = 0.02) ===")
for gm in [1, 2, 5, 10, 27.8, 50, 133.1]:
    print(f"gamma={gm:6.1f}  premium(rho=0.2)={100*prem_model(gm):6.2f}%  "
          f"premium(rho=1)={100*prem_model(gm,1.0):7.2f}%  rf={100*rf_model(gm):8.2f}%")

# --------------------------------------------- Figure 5.1: Hansen-Jagannathan
# Excess-return bound: sigma(m)/E[m] >= |E[Re]|/sigma(Re) = Sharpe ratio.
Em_grid = np.linspace(0.55, 1.03, 400)
bound = SR * Em_grid

def crra_moments(gamma, delta):
    Em = delta * np.exp(-gamma * g + 0.5 * gamma ** 2 * sig_c ** 2)
    Em2 = delta ** 2 * np.exp(-2 * gamma * g + 2 * gamma ** 2 * sig_c ** 2)
    return Em, np.sqrt(Em2 - Em ** 2)

def delta_matching_rf(gamma):
    """Time discount factor that reproduces the observed risk-free rate."""
    return (1.0 / (1 + rf_obs)) / np.exp(-gamma * g + 0.5 * gamma ** 2 * sig_c ** 2)


gam_pts = np.array([1, 5, 10, 15, 20, 25, 30, 40, 50], dtype=float)
pts_cal = np.array([crra_moments(gm, delta_matching_rf(gm)) for gm in gam_pts])
pts_fix = np.array([crra_moments(gm, 0.98) for gm in gam_pts])

gam_star = np.sqrt(np.log(1 + SR ** 2)) / sig_c
print(f"\ngamma satisfying the HJ bound exactly: {gam_star:.2f}")
print(f"check: E[m]={pts_cal[0,0]:.4f} (target {1/(1+rf_obs):.4f})")

fig, ax = plt.subplots(figsize=(4.9, 3.1))
ax.plot(Em_grid, bound, color="tab:blue", lw=1.2,
        label=r"Hansen--Jagannathan bound, $\sigma(m)=\mathrm{SR}\,\mathbb{E}[m]$")
ax.fill_between(Em_grid, bound, 0.9, color="tab:blue", alpha=0.08)
ax.plot(pts_cal[:, 0], pts_cal[:, 1], "o-", color="tab:red", ms=3.2, lw=0.9,
        label=r"CRRA, $\delta$ matched to $R_f$")
ax.plot(pts_fix[:, 0], pts_fix[:, 1], "s--", color="tab:green", ms=3.0, lw=0.9,
        label=r"CRRA, $\delta=0.98$ fixed")
for gm, (x, y) in zip(gam_pts, pts_cal):
    if gm in (1, 25, 50):
        ax.annotate(rf"$\gamma={int(gm)}$", (x, y), textcoords="offset points",
                    xytext=(6, -3), fontsize=7.5, color="tab:red")
for gm, (x, y) in zip(gam_pts, pts_fix):
    if gm in (50,):
        ax.annotate(rf"$\gamma={int(gm)}$", (x, y), textcoords="offset points",
                    xytext=(4, 2), fontsize=7.5, color="tab:green")
ax.set_xlim(0.55, 1.03); ax.set_ylim(0, 0.75)
ax.set_xlabel(r"$\mathbb{E}[m]$"); ax.set_ylabel(r"$\sigma(m)$")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig("fig_hj.pdf")

print("\n=== Table 5.2b CRRA volatility of the SDF ===")
for gm in [1, 5, 10, 20, 25.8, 30, 50]:
    Em, sm = crra_moments(gm, delta_matching_rf(gm))
    print(f"gamma={gm:6.1f}  E[m]={Em:.4f}  sigma(m)={sm:.4f}  ratio={sm/Em:.4f}  "
          f"(bound {SR:.4f})")

# ------------------------------------ Figure 5.2: premium and risk-free puzzle
gg = np.linspace(0, 60, 400)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
ax[0].plot(gg, 100 * prem_model(gg, rho_cm), color="tab:blue", lw=1.2,
           label=r"model premium, $\rho=0.20$")
ax[0].plot(gg, 100 * prem_model(gg, 1.0), color="tab:green", lw=1.0, ls="-.",
           label=r"model premium, $\rho=1$")
ax[0].axhline(100 * mu_e, color="tab:red", ls="--", lw=1.0, label="observed premium")
ax[0].set_xlabel(r"risk aversion $\gamma$"); ax[0].set_ylabel("equity premium (per cent)")
ax[0].set_ylim(0, 15); ax[0].legend(frameon=False, fontsize=7.5)
ax[1].plot(gg, 100 * rf_model(gg), color="tab:blue", lw=1.2, label="model risk-free rate")
ax[1].axhline(100 * rf_obs, color="tab:red", ls="--", lw=1.0, label="observed rate")
ax[1].set_xlabel(r"risk aversion $\gamma$"); ax[1].set_ylabel("risk-free rate (per cent)")
ax[1].set_ylim(-10, 80); ax[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig("fig_puzzle.pdf")

# ------------------------------------------- Section 6: simulated cross-section
rng = np.random.default_rng(20260817)
T_obs = 600                     # monthly observations, 50 years
n_assets = 10
lam1, lam2 = 0.0050, 0.0035     # monthly factor risk premia
s1, s2 = 0.0450, 0.0180         # factor volatilities
# orthogonal 5 x 2 design: five market loadings crossed with two second-factor loadings
b1 = np.repeat(np.array([0.70, 0.90, 1.10, 1.30, 1.50]), 2)
b2 = np.tile(np.array([-0.60, 0.60]), 5)
s_eps = 0.010

f1 = lam1 + s1 * rng.standard_normal(T_obs)
f2 = lam2 + s2 * rng.standard_normal(T_obs)
eps = s_eps * rng.standard_normal((T_obs, n_assets))
Re = b1 * (f1[:, None] - lam1) + b2 * (f2[:, None] - lam2) + b1 * lam1 + b2 * lam2 + eps

# true (population) values
true_mean = b1 * lam1 + b2 * lam2
true_beta_capm = (b1 * s1 ** 2) / s1 ** 2      # = b1 since factors independent
true_alpha = true_mean - true_beta_capm * lam1

# time-series CAPM regressions on the market factor only
X = np.column_stack([np.ones(T_obs), f1])
XtXi = np.linalg.inv(X.T @ X)
coef = XtXi @ X.T @ Re
resid = Re - X @ coef
s2res = (resid ** 2).sum(axis=0) / (T_obs - 2)
se = np.sqrt(np.outer(np.diag(XtXi), s2res))
alpha, beta = coef[0], coef[1]
t_alpha = alpha / se[0]

print("\n=== Table 6.1 simulated cross-section (seed 20260817, T=600) ===")
print(" i   b1     b2    E[Re]%   beta   alpha%   t(alpha)  true alpha%")
for i in range(n_assets):
    print(f"{i+1:2d} {b1[i]:5.2f} {b2[i]:6.2f} {100*Re[:,i].mean():7.3f} {beta[i]:6.3f} "
          f"{100*alpha[i]:7.3f} {t_alpha[i]:8.2f} {100*true_alpha[i]:11.3f}")

# GRS-style joint statistic
Sig_res = np.cov(resid.T, bias=True)
Sig_res += 1e-12 * np.eye(n_assets)
mu_f, var_f = f1.mean(), f1.var()
grs = (T_obs - n_assets - 1) / n_assets * (1 + mu_f ** 2 / var_f) ** -1 * \
      (alpha @ np.linalg.solve(Sig_res, alpha))
from scipy.stats import f as fdist
p_grs = 1 - fdist.cdf(grs, n_assets, T_obs - n_assets - 1)
print(f"GRS statistic = {grs:.2f},  p-value = {p_grs:.3e}")

# two-factor time-series regression: alphas should vanish
X2 = np.column_stack([np.ones(T_obs), f1, f2])
coef2 = np.linalg.inv(X2.T @ X2) @ X2.T @ Re
resid2 = Re - X2 @ coef2
s2res2 = (resid2 ** 2).sum(axis=0) / (T_obs - 3)
se2 = np.sqrt(np.outer(np.diag(np.linalg.inv(X2.T @ X2)), s2res2))
t_alpha2 = coef2[0] / se2[0]
print(f"two-factor model: max |alpha| = {100*np.abs(coef2[0]).max():.4f}%, "
      f"max |t| = {np.abs(t_alpha2).max():.2f}")

# cross-sectional regression of mean excess returns on CAPM betas
A = np.column_stack([np.ones(n_assets), beta])
gam_cs = np.linalg.lstsq(A, Re.mean(axis=0), rcond=None)[0]
r2 = 1 - ((Re.mean(axis=0) - A @ gam_cs) ** 2).sum() / \
     ((Re.mean(axis=0) - Re.mean()) ** 2).sum()
print(f"cross-sectional CAPM: intercept={100*gam_cs[0]:.4f}%, slope={100*gam_cs[1]:.4f}% "
      f"(true premium {100*lam1:.4f}%), R2={r2:.3f}")

A2 = np.column_stack([np.ones(n_assets), coef2[1], coef2[2]])
gam_cs2 = np.linalg.lstsq(A2, Re.mean(axis=0), rcond=None)[0]
r2b = 1 - ((Re.mean(axis=0) - A2 @ gam_cs2) ** 2).sum() / \
      ((Re.mean(axis=0) - Re.mean()) ** 2).sum()
print(f"cross-sectional 2-factor: slopes = {100*gam_cs2[1]:.4f}%, {100*gam_cs2[2]:.4f}% "
      f"(true {100*lam1:.4f}%, {100*lam2:.4f}%), R2={r2b:.3f}")

# ------------------------------------------------- Figure 6.1: security market line
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
bb = np.linspace(0.5, 1.5, 10)
ax[0].scatter(beta, 100 * Re.mean(axis=0), s=18, color="tab:blue", zorder=3,
              label="simulated portfolios")
ax[0].plot(bb, 100 * gam_cs[0] + 100 * gam_cs[1] * bb, color="tab:blue", lw=1.0,
           label="fitted cross-sectional line")
ax[0].plot(bb, 100 * lam1 * bb, ls="--", color="tab:red", lw=1.0,
           label="CAPM prediction")
ax[0].set_xlabel(r"CAPM beta $\beta_{i,\mathrm{mkt}}$")
ax[0].set_ylabel("mean excess return (per cent, monthly)")
ax[0].legend(frameon=False, fontsize=7.5)
ax[1].scatter(100 * (A @ gam_cs), 100 * Re.mean(axis=0), s=18, color="tab:blue",
              label="CAPM")
ax[1].scatter(100 * (A2 @ gam_cs2), 100 * Re.mean(axis=0), s=18, marker="s",
              facecolors="none", edgecolors="tab:red", label="two-factor model")
lims = [0, 100 * Re.mean(axis=0).max() * 1.15]
ax[1].plot(lims, lims, color="k", lw=0.8, ls=":")
ax[1].set_xlabel("predicted mean excess return (per cent)")
ax[1].set_ylabel("realised mean excess return (per cent)")
ax[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig("fig_sml.pdf")

# ------------------------------------------ Section 6: SDF Monte Carlo check
rng2 = np.random.default_rng(4242)
n_sim = 2_000_000
gamma_true, delta_true = 10.0, 0.98
Zc = rng2.standard_normal(n_sim)
Zi = rng2.standard_normal(n_sim)
dc = g + sig_c * Zc
r_m = rho_cm * sig_m * Zc + np.sqrt(1 - rho_cm ** 2) * sig_m * Zi
# choose the mean log return so that E[m R] = 1 exactly, then verify by simulation
mu_r = -np.log(delta_true) + gamma_true * g - 0.5 * gamma_true ** 2 * sig_c ** 2 \
       - 0.5 * sig_m ** 2 + gamma_true * rho_cm * sig_c * sig_m
R = np.exp(mu_r + r_m)
m = delta_true * np.exp(-gamma_true * dc)
pv = (m * R).mean()
se_pv = (m * R).std() / np.sqrt(n_sim)
print("\n=== Section 6.4 Monte Carlo check of E[mR]=1 (seed 4242, n=2,000,000) ===")
print(f"gamma={gamma_true}, delta={delta_true}: E[mR] = {pv:.6f} (s.e. {se_pv:.6f})")
print(f"implied risk-free rate = {100*(1/m.mean()-1):.3f}%  "
      f"(closed form {100*(np.exp(rf_model(gamma_true, -np.log(delta_true)))-1):.3f}%)")
Rf_cf = 1 / (delta_true * np.exp(-gamma_true * g + 0.5 * gamma_true ** 2 * sig_c ** 2))
prem_cf = np.exp(mu_r + 0.5 * sig_m ** 2) - Rf_cf
print(f"implied equity premium = {100*(R.mean()-1/m.mean()):.3f}% "
      f"(closed form {100*prem_cf:.3f}%)")
print(f"log premium E[r]-r_f+0.5 var = {100*(mu_r - np.log(Rf_cf) + 0.5*sig_m**2):.3f}% "
      f"(gamma*rho*sig_c*sig_m = {100*gamma_true*rho_cm*sig_c*sig_m:.3f}%)")
print("done")
