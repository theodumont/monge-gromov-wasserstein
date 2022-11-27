"""
The specific functions that reproduce results from the paper are:
- Algorithm 1, Figure 3: `grad_descent`
- Algorithm 2, Figure 4: `experiment_bimap`
- Figure 5: `exp_instability`
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import torch
import torchsort
from tqdm import tqdm
from itertools import groupby
from scipy.ndimage import binary_dilation
from matplotlib.colors import ListedColormap


# Optimal transport utils ================================================================

## Cost functions

def cost_xy(X, Y):
    """xy cost."""
    return X[:,None] @ Y[None,:]

def cost_x2y2(X, Y):
    """x^2y^2 cost."""
    return (X**2)[:,None] @ (Y**2)[None,:]

def cost_x2y2_p_4Mxy(X, Y, M0):
    """Linearized GW cost."""
    return cost_x2y2(X,Y) + 4 * M0 * cost_xy(X,Y)

def cost_GW(X, Y, pi):
    """GW cost."""
    return 2 * (cost_x2y2(X, Y) * pi).sum() + 4 * (cost_xy(X, Y)*pi).sum()**2

def cost_pi(pi):
    """GW cost."""
    x = pi[:,0]
    y = pi[:,1]
    n = len(x)
    return 2 * (x**2 * y**2).sum() / n + 4 * (x * y).sum()**2 / n**2

## Linear programs

def optim_max_M(X, Y, a, b):
    """
    Solve the linear program for m_max:

    primal: max_P     < C_{xy}, P >
    dual:   - max_{f,g} < a, f > + < b, g >
            s.t.        f + g <= - C_{xy}

    Parameters
    ----------
    X/Y : positions x_i/y_i
    a/b : weights for X,Y
    """
    n = len(X)
    m = len(Y)
    f = cp.Variable((n,1))
    g = cp.Variable((m,1))

    xy = cost_xy(X, Y)

    f_oplus_g = np.ones((n,1)) @ g.T + f @ np.ones((1,m))
    dual_cost = cp.sum(a.T @ f + b.T @ g)
    prob = cp.Problem(
        objective=cp.Maximize(dual_cost),
        constraints=[f_oplus_g <= -xy],
    )
    prob.solve(solver=cp.ECOS, verbose=False)
    return - prob.value

def optim_min_M(X, Y, a, b):
    """
    Solve the linear program for m_min:

    primal: min_P     < C_{xy}, P >
    dual:   max_{f,g} < a, f > + < b, g >
            s.t.      f + g <= C_{xy}

    Parameters
    ----------
    X/Y : positions x_i/y_i
    a/b : weights for X,Y
    """
    n = len(X)
    m = len(Y)
    f = cp.Variable((n,1))
    g = cp.Variable((m,1))

    xy = cost_xy(X, Y)

    f_oplus_g = np.ones((n,1)) @ g.T + f @ np.ones((1,m))
    dual_cost = cp.sum(a.T @ f + b.T @ g)
    prob = cp.Problem(
        objective=cp.Maximize(dual_cost),
        constraints=[f_oplus_g <= xy],
    )
    prob.solve(solver=cp.ECOS, verbose=False)
    return prob.value

def optim_linearized(X, Y, a, b, M0, _show, eps):
    """
    Solve the GW linearized linear program:

    primal: min_P     < -C_{x^2y^2} -4C_{xy}, P >
    dual:   max_{f,g} < a, f > + < b, g >
            s.t.      f + g <= -C_{x^2y^2+4Mxy}

    Parameters
    ----------
    X/Y : positions x_i/y_i
    a/b : weights for X,Y
    M0 : correlation / parameter for the linearized cost
    """
    n = len(X)
    m = len(Y)
    f = cp.Variable((n,1))
    g = cp.Variable((m,1))

    x2y2_p_4Mxy = cost_x2y2_p_4Mxy(X, Y, M0)

    f_oplus_g = np.ones((n,1)) @ g.T + f @ np.ones((1,m))
    dual_cost = cp.sum(a.T @ f + b.T @ g)
    prob = cp.Problem(
        objective=cp.Maximize(dual_cost),
        constraints=[f_oplus_g <= - x2y2_p_4Mxy],
    )
    prob.solve(solver=cp.ECOS)
    pi_disc, pi_cont = get_plan(prob, f, g, x2y2_p_4Mxy, M0, eps, _show)
    return pi_disc, pi_cont

def get_plan(prob, f, g, cost, M0, eps, _show):
    """
    Recover discrete and continuous transport plans from a cvxpy problem.

    Returns
    -------
    pi_disc : boolean transport plan recovered from f+g<c(x,y)
    pi_cont : primal variable of linear program
    """
    f = np.copy(f.value)
    g = np.copy(g.value)
    # discrete plan
    pi_disc = np.abs((f[:,None] + g[None,:])[:,:,0] + cost) < eps
    pi_disc = pi_disc.astype(float)
    # primal plan
    pi_cont = prob.constraints[0].dual_value
    if _show:
        plt.figure(figsize=(4,4))
        plt.imshow(pi_cont)
        plt.axis('off')
        setup_plot(title=f"Plan $\pi^\star_m$ for m={M0:.2f}", show=False, img=True)
    return pi_disc, pi_cont

## Input distributions

def make_beinert_distrib(n, eps=1e-2, a=1, b=1, c=1, d=1):
    """
    Return adversarial example from Beinert et al.

    Parameters
    ----------
    n : number of points
    eps : epsilon parameter from Beinert et al.
    """
    if eps>= 2/(n-3): print("Warning: not adversarial")
    def X_eps(eps):
        X = np.array(range(1, n+1))
        X = (2*X-n-1)/2*eps
        X[0] = -a
        X[-1] = b
        return X
    def Y_eps(eps):
        Y = np.array(range(1, n+1))
        Y = (Y-2)*eps
        Y[0] = -c
        Y[1] = -d+eps
        return Y
    X = X_eps(eps)
    Y = np.roll(Y_eps(eps), -1)
    pi_beinert = torch.tensor(np.stack((X, Y), axis=1), requires_grad=True)
    return X, Y, pi_beinert

def get_input(choice, pi_list=None):
    """
    Get input adversarial plan from string.

    Parameters
    ----------
    choice : type of adversarial plan
        * beinert7    : Beinert plan with 7 points
        * beinert15   : Beinert plan with 15 points
        * adversarial : adversarial plan from Algorithm 1 (GD)
        * random      : random plan
    """
    if choice   == 'beinert7':    pi_advers = make_beinert_distrib(n=7)[2]
    if choice   == 'beinert15':   pi_advers = make_beinert_distrib(n=15)[2]
    elif choice == 'adversarial': pi_advers = pi_list[-1]
    elif choice == 'random':      pi_advers = torch.rand(10,2)
    pi_advers = pi_advers.detach().numpy()
    pi_advers[:,0] -= pi_advers[:,0].mean()
    pi_advers[:,1] -= pi_advers[:,1].mean()
    assert np.allclose(pi_advers[:,0].mean(), 0)
    assert np.allclose(pi_advers[:,1].mean(), 0)
    return pi_advers



# Monge maps for the GW problem ==========================================================

## (Algorithm 1, Figure 3) Counter-examples for the optimality of monotone plans ---------

def objective(pi, sort_fun=lambda x: torchsort.soft_sort(x, regularization_strength=0.1)):
    """
    Objective functional c_GW(pi) - min{c_GW(pi_mon^+),c_GW(pi_mon^-)}.

    Parameters
    ----------
    sort_fun : differentiable sort function used to compute pi_mon^+ and pi_mon^-
    """
    # get marginals
    x = pi[:,0][None,:]
    y = pi[:,1][None,:]
    # build mon and monm
    pi_monP = torch.stack([sort_fun(x), sort_fun(y)], dim=2)[0]
    pi_monM = torch.stack([sort_fun(x), sort_fun(y).flip(dims=(0,1))], dim=2)[0]
    # cost
    cost_max_monP_monM = torch.max(cost_pi(pi_monP), cost_pi(pi_monM))
    objective = cost_max_monP_monM - cost_pi(pi)
    return objective

def grad_descent(pi, n_iter_max, eta, threshold, _verbose=False):
    """Algorithm 1. Gradient descent over the positions x_i and y_i.

    Parameters
    ----------
    pi : initial plan
    n_iter_max : maximum number of iterations of GD
    eta : step size
    threshold : early stopping threshold

    Returns
    -------
    [obj,pi,grad]_list : list of all objective values/plans/gradients across iterations
    """
    obj_list = []
    pi_list = []
    grad_list = []
    for i in range(n_iter_max):
        pi = pi - torch.mean(pi, 0)
        pi.retain_grad()
        obj = objective(pi)
        pi_list.append(pi.clone())
        obj_list.append(obj.clone().detach().numpy())
        if _verbose and i % max(1, n_iter_max // 15) == 0:
            print(f"[{i}] Objective = {obj}")
        obj.backward()
        with torch.no_grad():
            pi -= eta * pi.grad
        grad_list.append(pi.grad.clone())
        pi.grad.zero_()
        if obj_list[-1] > 1 or (i>=1 and obj_list[-1] > obj_list[-2]+.01) or (threshold is not None and obj_list[-1] < threshold):
            pi_list = pi_list[:-1]
            obj_list = obj_list[:-1]
            grad_list = grad_list[:-1]
            break
    return {
        'obj_list': obj_list,
        'pi_list': pi_list,
        'grad_list': grad_list,
        'n_iter_final': i,
    }

def experiment_GD(n, eta, n_iter_max, seed, threshold, init_GD, eps=1e-5, _show_obj=True, _verbose=False):
    """Wrapper for grad_descent.

    Parameters
    ----------
    n : number of points
    n_iter_max : maximum number of iterations of GD
    eta : step size
    threshold : early stopping threshold
    seed : random seed for reproducibility
    init_GD : type of initial plan
    """
    set_all_seeds(seed)
    if init_GD == "random":             pi_0 = torch.rand(n,2, requires_grad=True)
    elif init_GD == "beinert":          pi_0 = make_beinert_distrib(n=n)[2]
    res_GD = grad_descent(pi_0, eta=eta, n_iter_max=n_iter_max, threshold=threshold, _verbose=_verbose)
    obj_list = res_GD['obj_list']
    pi_list = res_GD['pi_list']
    grad_list = res_GD['grad_list']
    if _show_obj:
        plt.plot(obj_list)
        setup_plot(title="Objective function $c_{GW}(\pi) - \min\{c_{GW}(\pi_{mon}^+),c_{GW}(\pi_{mon}^-)\}$"+f" ($n={n}$)", xlabel="iterations", legend=False)
    is_adversarial = obj_list[-1] < - eps
    return {
        'obj_list': obj_list,
        'pi_list': pi_list,
        'grad_list': grad_list,
        'is_adversarial': is_adversarial,
        'n_iter_final': res_GD['n_iter_final'],
    }


## (Algorithm 2, Figure 4) Generating bi-maps from adversarial examples ------------------

def gauss(m, sig, n_pts, z_min, z_max):
    """Produce a Gaussian."""
    return np.exp(-(np.linspace(z_min, z_max, n_pts)-m)**2 / (2*sig**2)) / (sig * np.sqrt(2*np.pi))

def convolve(z, sigma, eps_val, n_pts):
    """Compute convolutions of sum of Dirac measures."""
    # extend domain
    z_min = np.min(z) - 4*sigma
    z_max = np.max(z) + 4*sigma
    # convolve
    conv_z = np.zeros(n_pts)
    for m in z:
        conv_z += gauss(m, sigma, n_pts, z_min, z_max)
    conv_z /= np.sum(conv_z)
    conv_z += eps_val
    conv_z /= np.sum(conv_z)
    domain = np.linspace(z_min, z_max, n_pts)
    return domain, conv_z

def get_nb_antecedents(P):
    """
    Count antecedents of '1's in the columns of the array P.

    Remarks
    -------
    * identical adjacent values are counted only once (itertools.groupby)
    * 'holes' of size 1 are filled (scipy.ndimage.binary_dilation)
    """
    antecedents = []
    for _, column in enumerate(P.T):
        column = binary_dilation(column).astype(column.dtype)
        antecedents.append(np.array([key for key, _ in groupby(column)]).sum())
    antecedents = np.array(antecedents)
    return antecedents

def is_bimap(pi_disc):
    """Compute if a plan is a bimap x->y or y->x.

    Parameters
    ----------
    pi_disc_list : discrete optimal plan

    Returns
    -------
    [xy]_bimap : boolean asserting if pi is a bimap
    [xy]_bimap_idx : indices of bimap coordinates for [xy]
    [xy]_antecedents : number of antecedents for each [xy]
    """
    x_antecedents = get_nb_antecedents(pi_disc)
    x_bimap_idx = np.where(x_antecedents >= 2)[0]
    y_antecedents = get_nb_antecedents(pi_disc.T)
    y_bimap_idx = np.where(y_antecedents >= 2)[0]

    x_bimap = len(x_bimap_idx) >= 2
    y_bimap = len(y_bimap_idx) >= 2
    return x_bimap, y_bimap, x_bimap_idx, y_bimap_idx, x_antecedents, y_antecedents

def show_plan_and_GW(x_bimap_idx, y_bimap_idx, x_antecedents, y_antecedents, pi_cont_list, M_list, costs_list, idx_max, x_new, y_new, _show_plan=True, _plot_GW=True):
    """Display optimal GW plan together with landscape of GW(pi^*_m)."""

    nb_cols = 1+int(_plot_GW)
    plt.subplots(2, nb_cols, figsize=(nb_cols*6,6), tight_layout=True)
    # plan -----------------------------------------
    plt.subplot(1, nb_cols, 1)
    pi = pi_cont_list[idx_max].T
    tmp = to_log(pi, min=1e-10)
    plt.imshow(tmp, cmap='Greys', origin='lower')
    plt.scatter(-1*np.ones_like(x_bimap_idx), x_bimap_idx, color="tab:green",  label="bi-map",      marker=",", s=len(y_antecedents)*.1, lw=0)
    plt.scatter(y_bimap_idx, -1*np.ones_like(y_bimap_idx), color="tab:orange", label="bi-anti-map", marker=",", s=len(x_antecedents)*.1, lw=0)
    # subdmod regions
    C = (cost_xy(x_new, y_new) * pi.T).sum()
    submod_true = (x_new[None,:]*y_new[:,None] <= -C)
    # plt.imshow(submod_true, cmap=ListedColormap(['none', 'tab:red']), origin='lower', alpha=.1)
    plt.imshow(1-submod_true, cmap=ListedColormap(['none', 'tab:green']), origin='lower', alpha=.1)
    setup_plot(title=f"Best plan $\pi^\star$", xlabel="X", ylabel="Y", spines_zero=True, show=False, legend=True)

    # landscape for GW(M) --------------------------
    if _plot_GW:
        # normal scale
        plt.subplot(2, nb_cols, nb_cols)
        plt.plot(M_list, -np.array(costs_list))
        plt.axvline(M_list[idx_max], color='r', linestyle="dotted")
        setup_plot(title="Landscape of GW$(\\pi^\star_m)$", xlabel="m", ylabel="GW cost", legend=False, show=False)
        # log scale
        plt.subplot(2, nb_cols, 2*nb_cols)
        plt.plot(M_list, -np.array(costs_list)+np.max(costs_list)+1e-10,color='purple')
        plt.semilogy()
        plt.axvline(M_list[idx_max], color='r', linestyle="dotted")
        setup_plot(title="Landscape of GW$(\\pi^\star_m)$ (log scale)", xlabel="m", ylabel="GW cost", legend=False, show=False)

    if _show_plan: plt.show()
    return plt

def experiment_bimap(pi_advers, n_pts, n_M, convolve_x=True, convolve_y=True, eps_val=1e-3, eps_plan=1e-6, sigma=5e-3, _show_convo=False, _show_plan=True, _plot_GW=True):
    """Algorithm 2. Generating bi-maps from adversarial examples.

    Parameters
    ----------
    pi_advers : adversarial plan
    n_pts : discretization precision (N_{\Delta x})
    n_M : discretization precision of the interval [m_min, m_max] (N_{\Delta m})
    convolve_[xy] : whether to convolve measure [xy] or to keep it a sum of Dirac measures
    sigma : standard deviation of convolution

    Returns
    -------
    pi_cont : optimal plan for GW
    """
    # getting marginals from pi
    x = pi_advers[:,0]
    y = pi_advers[:,1]

    # showing different values of variance
    if _show_convo:
        plt.subplots(2, 1, sharex=True, figsize=(12,8))
        for i, (z, name) in enumerate(zip([x, y], ["x", "y"])):
            plt.subplot(2, 1, i+1)
            for sigma_ in [.2, 1e-1, 5e-3]:
                plt.plot(*convolve(z, sigma=sigma_, eps_val=eps_val, n_pts=n_pts), label=f"$\\sigma={sigma_}$")
            plt.scatter(z, np.zeros_like(z), marker='x', alpha=.3, color="black", label=f"$\\pi_{name}$")
            setup_plot(title=f"Convolution of $\\pi_{name}$",show=False)
        plt.show()

    # sticking to a specific value for sigma and carrying on
    if convolve_x: x_new, a = convolve(x, sigma=sigma, eps_val=eps_val, n_pts=n_pts)
    else:         x_new, a = np.sort(x), np.ones_like(x) / x.shape[0]
    if convolve_y: y_new, b = convolve(y, sigma=sigma, eps_val=eps_val, n_pts=n_pts)
    else:         y_new, b = np.sort(y), np.ones_like(y) / y.shape[0]

    a = a[:,None]
    b = b[:,None]

    # bounds for M
    M_max = optim_max_M(x_new, y_new, a, b)
    M_min = optim_min_M(x_new, y_new, a, b)

    # testing range for M
    M_list = np.linspace(M_min, M_max, n_M)
    pi_disc_list = []
    pi_cont_list = []
    costs_list = []
    for i, M0 in tqdm(enumerate(M_list), total=n_M):
        pi_disc, pi_cont = optim_linearized(x_new, y_new, a, b, M0, eps=eps_plan, _show=False)
        pi_disc_list.append(pi_disc)
        pi_cont_list.append(pi_cont)
        costs_list.append(cost_GW(x_new, y_new, pi_cont))
    idx_max = np.argmax(costs_list)


    # bi-map?
    x_bimap, y_bimap, x_bimap_idx, y_bimap_idx, x_antecedents, y_antecedents = is_bimap(pi_disc_list[idx_max])
    if _show_plan: plot = show_plan_and_GW(x_bimap_idx, y_bimap_idx, x_antecedents, y_antecedents, pi_cont_list, M_list, costs_list, idx_max, x_new, y_new, _plot_GW=_plot_GW, _show_plan=_show_plan)
    else: plot = None
    return {
        "idx_max": idx_max,
        "pi_cont": pi_cont_list[idx_max],
        "pi_disc": pi_disc_list[idx_max],
        "pi_cont_list": pi_cont_list,
        "pi_disc_list": pi_disc_list,
        "M_list": M_list,
        "costs_list": costs_list,
        "x_new": x_new,
        "y_new": y_new,
        "a": a,
        "b": b,
        "x_bimap_idx": x_bimap_idx,
        "y_bimap_idx": y_bimap_idx,
        "x_antecedents": x_antecedents,
        "y_antecedents": y_antecedents,
        "plot": plot,
        "M_max": M_max,
        "M_min": M_min,
        "c_star": M_list[np.argmax(costs_list)],
    }


## (Figure 5) Instability of the optimality of monotone rearrangements -------------------

def exp_instability(pi_advers, sigma_list, n_pts=100, n_M=150, log_scale=False):
    """
    Figure 5. Instability of the optimality of monotone rearrangements.

    Parameters
    ----------
    pi_advers : adversarial plan
    sigma_list : list of standard deviations of convolutions
    n_pts : discretization precision (N_{\Delta x})
    n_M : discretization precision of the interval [m_min, m_max] (N_{\Delta m})
    """
    c_star_list = []
    plt.subplots(1, len(sigma_list), figsize=(len(sigma_list)*4,4), tight_layout=True)
    for i, sigma in enumerate(sigma_list):
        res = experiment_bimap(pi_advers, n_pts=n_pts, n_M=n_M, sigma=sigma, _show_convo=False, _show_plan=False)
        costs_list = res["costs_list"]
        M_list = res["M_list"]
        c_star_list.append(M_list[res["idx_max"]])
        plt.subplot(1, len(sigma_list), i+1)
        plt.plot(M_list, -np.array(costs_list))
        setup_plot(title=f"$\\sigma={sigma}_{i+1}$", xlabel="m", ylabel="GW cost", legend=False, show=False)
        if log_scale: plt.semilogy()
    plt.show()



# Utils ==================================================================================

## Display

def setup_plot(title=None, suptitle=None, xlabel=None, ylabel=None, legend=True, only_x=False, spines_zero=False, show=True, grid=True, img=False):
    """Util function for matplotlib.plt graphs."""
    if img: grid=False; legend=False
    if title    is not None: plt.title(title)
    if suptitle is not None: plt.suptitle(suptitle)
    if xlabel   is not None: plt.xlabel(xlabel)
    if ylabel   is not None: plt.ylabel(ylabel)
    if grid:                 plt.grid(ls=':')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if spines_zero:
        plt.gca().spines['bottom'].set_position('zero')
        plt.gca().spines['left'].set_position('zero')
    if only_x:
        plt.gca().spines['left'].set_visible(False)
        plt.yticks([])
    if img:
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.xticks([])
    if legend: plt.legend()
    if show:   plt.show()

def plot_pi(pi, subtitle="", _show=True):
    """Display a transport plan with a scatter plot."""
    plt.scatter(pi[:,0].detach().numpy(),pi[:,1].detach().numpy(), color="tab:blue")
    setup_plot(title=f"Plan $\\pi$ with $n={pi.shape[0]}$ points" + subtitle, legend=False, show=False)
    if _show: plt.show()

## Other

def to_log(img, min):
    """Transform image to log scale. Applied on primal variable of linear programs (transport plans).

    Parameters
    ----------
    img : image to transform
    min : minimum value to clip img with
    """
    max = img.max()
    return (np.log(img.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))

## Randomness

def set_all_seeds(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True