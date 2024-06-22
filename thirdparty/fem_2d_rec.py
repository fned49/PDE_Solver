import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import math

def mesh_fem_2d_rec(xl, xr, yl, yr, Mx, My, k):
    xx = np.linspace(xl, xr, k*Mx+1)
    yy = np.linspace(yl, yr, k*My+1)
    c4n = np.array([[x, y] for y in yy for x in xx])

    Nx = k*Mx+1
    Ny = k*My+1
    ind4e = np.array([[(j+k*m)*Nx+(i+k*n) for j in range(k+1) for i in range(k+1)] for m in range(My) for n in range(Mx)])
    n4e = ind4e[:, [0, k, (k+1)**2-1, k*(k+1)]]

    nNodes = Nx*Ny
    n4db = np.array([[[k*i+j for j in range(k+1)] for i in range(Mx)],
          [[k*Nx*(i) +k*Mx + j*Nx for j in range(k+1)] for i in range(My)],
          [[(nNodes - 1) - k*i - j for j in range(k+1)] for i in range(Mx)],
          [[Nx*(Ny-1) - k*Nx*i - j*Nx for j in range(k+1)] for i in range(My)]]).reshape(-1, k+1)
    return (c4n, n4e, n4db, ind4e)

def fem_poisson_solver(c4n, n4e, n4db, ind4e, k, M_R, Srr_R, Sss_R, f, u_D):
    number_of_nodes = c4n.shape[0]
    number_of_elements = n4e.shape[0]
    b = np.zeros(number_of_nodes)
    u = np.zeros(number_of_nodes)
    xr = np.array([(c4n[n4e[i, 1], 0] - c4n[n4e[i, 0], 0]) / 2 for i in range(number_of_elements)])
    ys = np.array([(c4n[n4e[i, 3], 1] - c4n[n4e[i, 0], 1]) / 2 for i in range(number_of_elements)])
    J = xr * ys
    rx = ys / J
    sy = xr / J
    Aloc = np.array([J[i] * (rx[i]**2 * Srr_R.flatten() + sy[i]**2 * Sss_R.flatten()) for i in range(number_of_elements)])
    for i in range(number_of_elements):
        b[ind4e[i]] += J[i] * np.matmul(M_R, f(c4n[ind4e[i]]))
    row_ind = np.tile(ind4e.flatten(), ((k+1)**2, 1)).T.flatten()
    col_ind = np.tile(ind4e, (1, (k+1)**2)).flatten()
    A_COO = coo_matrix((Aloc.flatten(), (row_ind, col_ind)), shape=(number_of_nodes, number_of_nodes))
    A = A_COO.tocsr()
    dof = np.setdiff1d(range(0, number_of_nodes), np.unique(n4db))
    u[dof] = spsolve(A[dof, :].tocsc()[:, dof].tocsr(), b[dof])
    return u

def get_matrices_2d(k=1):
    r = np.linspace(-1, 1, k+1)
    V = VandermondeM1D(k, r)
    D_R_1d = Dmatrix1D(k, r, V)
    invV = np.linalg.solve(V, np.identity(k+1))

    M_R_1d = invV.T @ invV
    M_R = np.kron(M_R_1d, M_R_1d)

    Dr_R = np.kron(np.identity(k+1), D_R_1d)
    Ds_R = np.kron(D_R_1d, np.identity(k+1))

    Srr_R = Dr_R.T @ M_R @ Dr_R
    Sss_R = Ds_R.T @ M_R @ Ds_R
    return (M_R, Srr_R, Sss_R, Dr_R, Ds_R)

def nJacobiP(x, alpha=0, beta=0, degree=0):
    Pn = np.zeros((degree+1, x.size), float)
    Pn[0, :] = np.sqrt(2.0 ** (-alpha - beta - 1) \
                       * math.gamma(alpha + beta + 2) \
                       / ((math.gamma(alpha + 1) * math.gamma(beta + 1))))

    if degree == 0:
        P = Pn
    else:
        Pn[1, :] = np.multiply(Pn[0, :] * np.sqrt((alpha + beta + 3.0) \
            / ((alpha + 1) * (beta + 1))), ((alpha + beta + 2) * x + (alpha - beta))) / 2
        a_n = 2.0 / (2 + alpha + beta) * np.sqrt((alpha + 1.0) * (beta + 1.0) \
                                                / (alpha + beta + 3.0))
        for n in range(2, degree + 1):
            anew = 2.0 / (2 * n + alpha + beta) * np.sqrt(n * (n + alpha + beta) \
                                                          * (n + alpha) * (n + beta) \
                                                          / ((2 * n + alpha + beta - 1.0) * (2 * n + alpha + beta + 1.0)))
            b_n = -(alpha ** 2 - beta ** 2) \
                / ((2 * (n - 1) + alpha + beta) * (2 * (n - 1) + alpha + beta + 2.0))
            Pn[n, :] = (np.multiply((x - b_n), Pn[n - 1, :]) - a_n * Pn[n - 2, :]) \
                       / anew
            a_n = anew

    P = Pn[degree, :]
    return P

def VandermondeM1D(degree, r):
    V1D = np.zeros((r.size, degree + 1), float)
    for j in range(0, degree + 1):
        V1D[:, j] = nJacobiP(r, 0, 0, j)
    return V1D

def DnJacobiP(x, alpha=0, beta=0, degree=0):
    dP = np.zeros(x.size, float)
    if degree == 0:
        dP[:] = 0
    else:
        dP[:] = np.sqrt(degree * (degree + alpha + beta + 1.0)) \
                * nJacobiP(x, alpha + 1, beta + 1, degree - 1)
    return dP

def Dmatrix1D(degree, r, V):
    Vr = DVandermondeM1D(degree, r)
    Dr = np.linalg.solve(np.transpose(V), np.transpose(Vr))
    Dr = np.transpose(Dr)
    return Dr

def DVandermondeM1D(degree, r):
    DVr = np.zeros((r.size, degree + 1), float)
    for j in range(0, degree + 1):
        DVr[:, j] = DnJacobiP(r, 0, 0, j)
    return DVr