"""
SIMODS deterministic limitation paper — Phase 2 clean vehicle.

Trains a DIRECT MSE regressor x_obs -> theta=(S_I, sigma) on synthetic OGTT.
This is the honest vehicle for Prop 3 (MSE-optimal estimator = conditional mean):
it has no fixed-point pathology (unlike the iterative operator, cf. G0), so it
should collapse cleanly onto the off-fiber conditional-mean locus (Thm 1).

Outputs -> results/det_meanlocus/ : predictions.npz, metrics.json, figure.
"""
import numpy as np, torch, torch.nn as nn, json, os
np.random.seed(0); torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = 'results/det_meanlocus'; os.makedirs(OUT, exist_ok=True)

# ---- data: glucose-only observations -> (S_I, sigma) ----
d = np.load('data/ogtt_simul/augmented_data_ode_noderiv_100000.npz')
X = d['observed_data'].reshape(d['observed_data'].shape[0], -1).astype(np.float32)  # (N,5)
P = d['params'].astype(np.float32)                                                  # (N,2)
N = len(X)
idx = np.random.permutation(N)
ntr = int(0.85 * N)
tr, te = idx[:ntr], idx[ntr:]

# observed: standardize on train; params: log + min-max to [-1,1] on train (repo scheme)
xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-8
Xn = (X - xm) / xs
logP = np.log(np.clip(P, 1e-8, None))
pmin, pmax = logP[tr].min(0), logP[tr].max(0)
Pn = (2 * (logP - pmin) / (pmax - pmin + 1e-8) - 1).astype(np.float32)
def denorm(pn):
    lp = (pn + 1) / 2 * (pmax - pmin) + pmin
    return np.exp(lp)

Xtr = torch.tensor(Xn[tr], device=dev); Ptr = torch.tensor(Pn[tr], device=dev)
Xte = torch.tensor(Xn[te], device=dev)

class Reg(nn.Module):
    def __init__(s, h=256):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(5, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                              nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2), nn.Tanh())
    def forward(s, x): return s.net(x)

m = Reg().to(dev)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
lossf = nn.MSELoss()
nb = 512
for ep in range(400):
    perm = torch.randperm(len(Xtr), device=dev)
    for i in range(0, len(Xtr), 8192):
        b = perm[i:i+8192]
        opt.zero_grad(); l = lossf(m(Xtr[b]), Ptr[b]); l.backward(); opt.step()
    if ep % 50 == 0 or ep == 399:
        with torch.no_grad(): tl = lossf(m(Xtr), Ptr).item()
        print(f"ep {ep:3d}  train MSE(norm) {tl:.4f}", flush=True)

with torch.no_grad():
    Pte_pred = denorm(m(Xte).cpu().numpy())
Pte_true = P[te]
np.savez(f'{OUT}/predictions.npz', p_true=Pte_true, p_pred=Pte_pred)

# ---- evaluation: collapse to off-fiber conditional-mean locus ----
pt, pp = Pte_true.astype(float), Pte_pred.astype(float)
C = pt[:, 0] * pt[:, 1]
nb = 20; qs = np.quantile(C, np.linspace(0, 1, nb + 1)); qs[-1] += 1e-9
bid = np.clip(np.digitize(C, qs) - 1, 0, nb - 1)
ta, pa, dmean, dmid, jof, mdi = [], [], [], [], [], []
for b in range(nb):
    mm = bid == b
    if mm.sum() < 15: continue
    ta.append(np.std(np.log(pt[mm, 0]))); pa.append(np.std(np.log(np.clip(pp[mm, 0], 1e-6, None))))
    mC = pt[mm].mean(0); pc = pp[mm].mean(0); cc = C[mm].mean()
    dmean.append(np.linalg.norm(pc - mC)); dmid.append(np.linalg.norm(pc - np.array([np.sqrt(cc), np.sqrt(cc)])))
    jof.append(mC[0] * mC[1] - cc); mdi.append(abs(pc[0] * pc[1] - cc) / cc)
ta, pa = np.array(ta), np.array(pa)
C_pred = pp[:, 0] * pp[:, 1]
res = dict(
    collapse_ratio=float(pa.mean() / ta.mean()), true_along=float(ta.mean()), pred_along=float(pa.mean()),
    jensen_offfiber_mean=float(np.mean(jof)), jensen_frac_pos=float(np.mean(np.array(jof) > 0)),
    mDI_relerr=float(np.mean(np.abs(C_pred - C) / C)), mDI_relerr_inrange=float(np.mean(mdi)),
    selector_dist_condmean=float(np.mean(dmean)), selector_dist_midpoint=float(np.mean(dmid)),
    selector=('CONDITIONAL-MEAN' if np.mean(dmean) < np.mean(dmid) else 'MIDPOINT/other'),
    n_test=int(len(te)),
)
json.dump(res, open(f'{OUT}/metrics.json', 'w'), indent=2)
print("\n=== DIRECT REGRESSOR (clean vehicle) ===")
for k, v in res.items(): print(f"  {k}: {v}")
PY_DONE = True
