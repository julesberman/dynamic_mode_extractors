#%%

from matplotlib import pyplot as plt
import numpy as np, scipy.linalg, scipy
import matplotlib.pylab as pl


def gen_results(eigs, coeffs, noise, reg):

    ord = len(eigs)
    eigs = np.array(eigs)
    coeffs = np.array(coeffs)
    VdM = np.concatenate([[eigs ** i] for i in range(ord)])
    VdMinv = np.linalg.inv(VdM)
    A = VdM @ np.diag(eigs) @ VdMinv

    x_func = lambda t: coeffs @ (eigs ** t)
    x_mat = np.array([[x_func(t + t0) for t0 in range(ord)] for t in range(ord)])

    Cx = x_mat.T @ x_mat / ord

    S = np.zeros([ord, ord])
    S[:-1, 1:] = np.eye(ord - 1)

    Id = np.eye(ord)
    Anoisy = (A @ Cx + noise ** 2 * S) @ np.linalg.inv(Cx + (noise ** 2 + reg) * Id)

    eigvals, lefts, rights = scipy.linalg.eig(Anoisy, left=True)
    eigs_ord = np.argsort(eigvals.real)[::-1]

    # lefts /= lefts[-1:, :]
    lefts /= [[np.linalg.norm(lefts[:, i]) for i in range(ord)]]
    lefts /= np.sign(lefts[-1:, :])
    rights /= rights[:1, :]

    RIF = np.real_if_close

    return RIF(eigvals[eigs_ord]), RIF(lefts.T[eigs_ord]), RIF(rights.T[eigs_ord])


# %%
eigs = [2, 1.2, 0.9, 0.7, 0.2]
coeffs = [0.3, 0.5, 1.0, 1.5, 3.0]
reg = 0
ord = len(eigs)

noises = np.logspace(-5.9, 1.3, 30)
colors = pl.cm.jet(np.linspace(0, 1, len(noises)))

plt.figure(figsize=(5, 4))
plt.axhline(0, color="k", lw=0.5, dashes=[10, 5])
for i, noise in enumerate(noises):
    lefts = gen_results(eigs, coeffs, noise, reg=reg)[1]
    plt.plot(
        lefts[0].real,
        label="noise = {:.0e}".format(noise) if noise > 0 else "noise = 0",
        color=colors[i],
        alpha=0.4,
    )
plt.xticks(range(ord), ["$t - {}$".format(ord - i) for i in range(ord)])
plt.title("Top left eigenvector")
plt.tight_layout()
plt.savefig("left_anal.pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(5, 4))
plt.axhline(0, color="k", lw=0.5, dashes=[10, 5])
for i, noise in enumerate(noises):
    rights = gen_results(eigs, coeffs, noise, reg=reg)[2]
    plt.plot(
        rights[0].real,
        label="noise = {:.0e}".format(noise) if noise > 0 else "noise = 0",
        color=colors[i],
        alpha=0.4,
    )
plt.xticks(range(ord), ["$t - {}$".format(ord - i) for i in range(ord)])
plt.title("Top left eigenvector")
plt.tight_layout()
plt.savefig("right_anal.pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(3.5, 4))
plt.axhline(0, color="k", lw=0.5, dashes=[10, 5])
for i, noise in enumerate(noises):
    eigvals = gen_results(eigs, coeffs, noise, reg=reg)[0]
    plt.plot(
        eigvals.real,
        label="noise = {:.0e}".format(noise) if noise > 0 else "noise = 0",
        color=colors[i],
        alpha=0.4,
    )
plt.xticks(range(ord), 1 + np.arange(ord))
plt.title("Eigenvalues (real part)")
plt.tight_layout()
plt.savefig("eigs_real_anal.pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(3.5, 4))
plt.axhline(0, color="k", lw=0.5, dashes=[10, 5])
for i, noise in enumerate(noises):
    eigvals = gen_results(eigs, coeffs, noise, reg=reg)[0]
    plt.plot(
        eigvals.imag,
        label="noise = {:.0e}".format(noise) if noise > 0 else "noise = 0",
        color=colors[i],
        alpha=0.4,
    )
plt.xticks(range(ord), 1 + np.arange(ord))
plt.title("Eigenvalues (imaginary part)")
plt.tight_layout()
plt.savefig("eigs_imag_anal.pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(0.7, 4))
for i, noise in enumerate(noises):
    plt.axhline(noise, color=colors[i])
plt.yscale("log")
plt.xticks([])
plt.title("noise amount")
# plt.tight_layout()
plt.savefig("bar_anal.pdf", bbox_inches="tight")
plt.show()
