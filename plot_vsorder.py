# %%

import numpy as np, matplotlib.pyplot as plt, scipy, scipy.linalg, pandas as pd, seaborn as sns


def nan_argsort(a):
    temp = a.copy()
    temp[np.isnan(a)] = -np.inf
    return temp.argsort()


Exps = np.sort(np.array([1.5, 0.4, -0.15, -0.8, -1.5, -2.5]))

N_max = 9
Ns = np.arange(3, N_max + 1, 3)
num_runs = 20
T = np.int(2 * N_max)
shifts = np.arange(1, 11, 3)
shift_max = max(shifts)

time = np.arange(0, 2.55, 0.1)


noise = 0
results_eval = {
    "eval_r": [],
    "eval_im": [],
    "noise": [],
    "order": [],
    "order_num": [],
    "sysord": [],
    "run": [],
}

order_dict = {0: "1st", 1: "2nd", 2: "3rd"}
for i in range(3, 20):
    order_dict[i] = str(i + 1) + "th"
results_evec = {
    "evec_r": [],
    "evec_im": [],
    "noise": [],
    "component": [],
    "sysord": [],
    "run": [],
}
N = 9
sysords = [2, 4, 6]
noises = [
    0.00000001,
    0.0000001,
    0.000001,
    0.00001,
    0.0001,
    0.0003,
    0.001,
    0.01,
    0.03,
    0.1,
]
for sysord in sysords:
    for noise in noises:
        for run in range(50):

            Inits = np.array([0.01, 0.05, 0.2, 1, 0.2, 0.5][:sysord])[::-1]
            Xseries = np.exp(
                time.reshape(-1, 1) * Exps[-sysord:].reshape(1, -1)
            ) * Inits.reshape(1, -1)

            XseriesNoisy = Xseries * (1 + noise * np.random.randn(*Xseries.shape))
            XsereisTotal = XseriesNoisy.sum(1)

            shift = 1
            X = scipy.linalg.hankel(XsereisTotal)[: N + shift, : -N - shift + 1]

            X0 = X[:-shift]
            Xp = X[shift:]
            X0Xp = X0 @ Xp.T / X0.shape[1]
            X0X0 = X0 @ X0.T / X0.shape[1]

            GEV_sol = scipy.linalg.eig(X0Xp, X0X0)
            evals = np.real_if_close(GEV_sol[0])
            evecs = np.real_if_close(GEV_sol[1])

            sort_ord = nan_argsort(np.real(evals))[::-1]
            for i, ord in enumerate(sort_ord):
                if i > 8:
                    continue
                results_eval["eval_r"].append(np.real(evals[ord]))
                results_eval["eval_im"].append(np.imag(evals[ord]))
                results_eval["noise"].append(noise)
                results_eval["order"].append(order_dict[i])
                results_eval["order_num"].append(i + 1)
                results_eval["sysord"].append(sysord)
                results_eval["run"].append(run)

            sign = np.sign(np.real(evecs[-1, sort_ord[0]]))
            for i, el in enumerate(evecs[:, sort_ord[0]]):
                results_evec["noise"].append(noise)
                results_evec["component"].append(-N + i)
                results_evec["sysord"].append(sysord)
                results_evec["evec_r"].append(np.real(sign * el))
                results_evec["evec_im"].append(np.imag(sign * el))
                results_evec["run"].append(run)


noise = 0.03

Xseries = np.exp(time.reshape(-1, 1) * Exps.reshape(1, -1)) * Inits.reshape(1, -1)
XseriesNoisy = Xseries * (1 + noise * np.random.randn(*Xseries.shape))
XsereisTotal = XseriesNoisy.sum(1)

[
    plt.plot(time, XseriesNoisy[:, i], lw=1.5, label="$x_{}(t)$".format(i))
    for i in range(len(Exps))
]
plt.plot(time, XsereisTotal, label="total", color="black")
plt.title("Series constituents and total")
plt.axhline(0, color="black", lw=0.5, dashes=[10, 8])
plt.legend(loc=1)
plt.xlabel("time (s)", fontsize=14)
plt.ylabel("series", fontsize=14)
plt.savefig("series_6o.pdf")
plt.show()

results_eval_df = pd.DataFrame(results_eval)
results_eval_df = results_eval_df[
    ~results_eval_df.isin([np.nan, np.inf, -np.inf]).any(1)
]
results_evec_df = pd.DataFrame(results_evec)
#%%

plt.axhline(0, color="k", lw=0.8)
sns.set_theme(style="whitegrid")
palette = sns.color_palette("tab10", 3)
paper_rc = {"lines.linewidth": 1.0, "lines.markersize": 10}
sns.set_context("paper", rc=paper_rc)
sns.pointplot(
    data=results_evec_df.query("noise== 1e-4"),
    y="evec_r",
    hue="sysord",
    x="component",
    palette=palette,
    legend=False,
    lw=1.13,
)
plt.xlabel("lag time", fontsize=14)
plt.ylabel("", fontsize=14)
# plt.title("Top left Eigenvector", fontsize=14)
legend = plt.legend(ncol=3, title="system order ($k$)", loc=2, fontsize=13)
plt.setp(legend.get_title(), fontsize=13)
plt.yticks([-0.6, -0.3, 0, 0.3, 0.6], fontsize=11)
plt.text(
    -0.16, 0.92, "B", fontweight="bold", fontsize="28", transform=plt.gca().transAxes
)
plt.savefig("vsorder.pdf")
# %%
