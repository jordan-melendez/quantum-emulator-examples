import numpy as np
from numpy.typing import ArrayLike
from numpy import ndarray
from nptyping import NDArray
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from typing import Optional, Any

# Purely for type hints & documentation
# OneDArray = NDArray[(Any,)]
# TwoDArray = NDArray[(Any, Any)]
# ThreeDArray = NDArray[(Any, Any, Any)]


class BoundStateHamiltonian:
    def __init__(self, name: str, H0, H1):
        self.name = name
        self.H0 = H0
        self.H1 = H1
        self.H0_sub = None
        self.H1_sub = None
        self.X = None
        self.N = None
        self.p_train = None
        self.E_train = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    def compute_full_hamiltonian(self, p):
        return self.H0 + self.H1 @ p

    def compute_subspace_hamiltonian(self, p):
        return self.H0_sub + self.H1_sub @ p

    def solve_schrodinger_exact(self, p):
        H = self.compute_full_hamiltonian(p)
        E, psi = eigh(H, subset_by_index=[0, 0])
        return E[0], psi[:, 0]

    def solve_schrodinger_subspace(self, p):
        H_sub = self.compute_subspace_hamiltonian(p)
        E, beta = eigh(H_sub, b=self.N, type=1, subset_by_index=[0, 0])
        return E[0], beta[:, 0]

    def exact_wave_function(self, p):
        return self.solve_schrodinger_exact(p)[1]

    def coefficients(self, p):
        return self.solve_schrodinger_subspace(p)[1]

    def emulate_wave_function(self, p):
        return self.X @ self.coefficients(p)

    def fit(self, p_train):
        """

        Parameters
        ----------
        p_train

        Returns
        -------

        """

        # Create subspace from exact solutions
        X = []
        E_train = np.zeros(len(p_train))
        for i, p in enumerate(p_train):
            E, psi = self.solve_schrodinger_exact(p)
            E_train[i] = E
            X.append(psi)
        X = np.stack(X, axis=1)

        self.setup_projections(X)
        self.p_train = p_train
        self.E_train = E_train
        return self

    def setup_projections(self, X):
        N = X.T @ X

        # Project matrices once and store them
        H0_sub = X.T @ self.H0 @ X
        H1_sub_reshaped = X.T @ (np.transpose(self.H1, (2, 0, 1)) @ X)
        H1_sub = np.transpose(H1_sub_reshaped, (1, 2, 0))

        self.X = X
        self.N = N
        self.H0_sub = H0_sub
        self.H1_sub = H1_sub
        return self

    def predict(self, p, use_emulator=True):
        if use_emulator:
            E, beta = self.solve_schrodinger_subspace(p)
            return E
        else:
            E, psi = self.solve_schrodinger_exact(p)
            return E


class BoundStateOperator:
    def __init__(
        self,
        name: str,
        ham: BoundStateHamiltonian,
        op0,
        op1=None,
        ham_right: Optional[BoundStateHamiltonian] = None,
    ):
        self.name = name

        self.ham_left = ham
        self.ham_right = ham_right
        self.p_train = self.ham_left.p_train

        if ham_right is not None:
            self._transition = True
        else:
            self._transition = False

        X_left = X_right = self.ham_left.X
        if self._transition:
            X_right = self.ham_right.X

        op0_sub = X_left.T @ op0 @ X_right
        if op1 is not None:
            op1_t = np.transpose(op1, axes=(2, 0, 1))
            op1_sub = X_left.T @ (op1_t @ X_right)
            op1_sub = np.transpose(op1_sub, axes=(1, 2, 0))
        else:
            op1_sub = None

        self.op0 = op0
        self.op1 = op1
        self.op0_sub = op0_sub
        self.op1_sub = op1_sub

    def __repr__(self):
        if self._transition:
            return f'<{self.ham_left.name} | {self.name} | {self.ham_right.name}>'
        else:
            return f'<{self.ham_left.name} | {self.name} | {self.ham_left.name}>'

    def compute_full_operator(self, p):
        op = self.op0
        if self.op1 is not None:
            op = op + self.op1 @ p
        return op

    def compute_subspace_operator(self, p):
        op = self.op0_sub
        if self.op1_sub is not None:
            op = op + self.op1_sub @ p
        return op

    def compute_full_gs_psi_left_and_right(self, p):
        psi_left = self.ham_left.exact_wave_function(p)
        if self._transition:
            psi_right = self.ham_right.exact_wave_function(p)
        else:
            psi_right = psi_left
        return psi_left, psi_right

    def compute_gs_beta_left_and_right(self, p):
        beta_left = self.ham_left.coefficients(p)
        if self._transition:
            beta_right = self.ham_right.coefficients(p)
        else:
            beta_right = beta_left
        return beta_left, beta_right

    def expectation_value_exact(self, p):
        psi_left, psi_right = self.compute_full_gs_psi_left_and_right(p)
        op = self.compute_full_operator(p)
        return psi_left.T @ op @ psi_right

    def expectation_value_emulator(self, p):
        r"""

        The beta are normalized so no normalization of this quantity
        should be necessary.

        Parameters
        ----------
        p

        Returns
        -------

        """
        beta_left, beta_right = self.compute_gs_beta_left_and_right(p)
        op = self.compute_subspace_operator(p)
        return beta_left.T @ op @ beta_right

    def predict(self, p, use_emulator=True):
        R"""Computes the operator expectation value between states of the left and right Hamiltonian

        Uses eigenvector continuation for speed

        Parameters
        ----------
        p : array, shape = (n_p,)
            The Hamiltonian and Operator parameters at which to compute the expectation value
        use_emulator : bool, optional
            Whether to use the emulator or compute exactly. Defaults to True.

        Returns
        -------
        float
            The expectation value
        """
        if use_emulator:
            return self.expectation_value_emulator(p)
        else:
            return self.expectation_value_exact(p)
