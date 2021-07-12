import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_genlaguerre, gammaln
from scipy.integrate import quadrature

from .constants import hbar_c, pi


def leggauss_shifted(deg, a=-1, b=1):
    """Obtain the Gaussian quadrature points and weights when the limits of integration are [a, b]

    Parameters
    ----------
    deg : int
        The degree of the quadrature
    a : float
        The lower limit of integration. Defaults to -1, the standard value.
    b : float
        The upper limit of integration. Defaults to +1, the standard value.

    Returns
    -------
    x : The integration locations
    w : The weights
    """
    x, w = leggauss(deg)
    w *= (b - a) / 2.0
    x = ((b - a) * x + (b + a)) / 2.0
    return x, w


def compute_omega(mass, b):
    R"""Returns omega in units of MeV

    Parameters
    ----------
    mass
    b
    """
    return hbar_c ** 2 / (mass * b ** 2)


def ho_radial_wf(r, n, ell, b):
    r"""The radial wave function u_{nl} for the 3d isotropic harmonic oscillator.

    These are normalized such that \int |u_nl(r)|**2 dr = 1

    Parameters
    ----------
    r :
        The distance in fm
    n :
        The n quantum number
    ell :
        The angular momentum quantum number
    b :
        The oscillator parameter

    Returns
    -------
    u_nl
    """
    # b = 1 / np.sqrt(mass * omega / hbar_c)
    # N_{nl} = 2 Gamma(n) / [b * Gamma(n + l + 1/2)]
    norm = np.sqrt(2 * np.exp(gammaln(n) - np.log(b) - gammaln(n + ell + 0.5)))
    y = r / b
    y2 = y ** 2
    laguerre = eval_genlaguerre(n - 1, ell + 0.5, y2)
    return norm * y ** (ell + 1) * np.exp(-y2 / 2) * laguerre


# def ho_potential(r, )


def ho_energy(n, ell, omega):
    R"""The energy of the harmonic oscillator

    Note that N = 2 (n - 1) + ell.

    Parameters
    ----------
    n
    ell
    omega

    Returns
    -------

    """
    return omega * (2 * (n - 1) + ell + 3 / 2)


def kinetic_energy_ho_basis(n_max, omega=1):
    n = np.arange(0, n_max+1)
    n_up = n[1:]
    n_lo = n[:-1]
    p2 = np.diag(2 * n + 1.5)
    p2[n_up - 1, n_up] = -np.sqrt(n_up * (n_up + 1.5))
    p2[n_lo + 1, n_lo] = -np.sqrt((n_lo + 1) * (n_lo + 1.5))
    return omega * p2 / 2.0


# def kinetic_energy_momentum_basis(n_max, omega=1):
#     return np.diag


def convert_from_r_to_ho_basis(v, n_max, ell, r, dr, b):
    wfs_ho = np.stack([ho_radial_wf(r, n=i+1, ell=ell, b=b) for i in range(n_max+1)])
    out = np.zeros((n_max + 1, n_max + 1))
    for n in range(n_max + 1):
        u_nl = wfs_ho[n]
        for m in range(n, n_max + 1):
            u_ml = wfs_ho[m]
            out[n, m] = out[m, n] = np.sum(v * u_nl * u_ml * dr)
    return out


def fourier_transform_spherical(v, k, ell, r, dr):
    from scipy.special import spherical_jn
    j_l = np.sqrt(2 / pi) * (r * np.sqrt(dr)) * spherical_jn(ell, np.outer(k, r))
    return (j_l[:, None, :] * j_l[None, :, :]) @ v


class SphericalWell:
    def __init__(self, V0, R=1.0, mu=1.0, hbar=1.0, r_max=20.0, gauss_pts=48):
        self.V0 = V0
        self.R = R
        self.mu = mu
        self.hbar = hbar
        self.r_max = r_max
        r, dr = leggauss_shifted(gauss_pts, 0, r_max)
        self.r = r
        self.dr = dr

        self.E0 = None
        self.k_gs = None
        self.kappa_gs = None
        self.norm_const = None

        self.setup(V0=V0, R=R)

    def potential(self, r, Vd=None):
        """
        Potential for the square well.
        Defined so that it works with scalar or numpy vector r.
        """
        if Vd is None:
            Vd = self.V0
        ans = np.where(r > self.R, 0.0, -Vd)
        return ans

    def setup(self, V0=None, R=None):
        if V0 is not None:
            self.V0 = V0
        if R is not None:
            self.R = R
        self.E0 = self.find_E0()
        self.k_gs = self.k_in(self.E0)
        self.kappa_gs = self.kappa_out(self.E0)

        u_sq = self.unnormalized_wave_function(self.r) ** 2
        self.norm_const = (self.dr @ u_sq) ** (-0.5)
        return self

    def k_in(self, E):
        """
        Local wave number inside square-well potential
        """
        return np.sqrt(2.0 * self.mu * (self.V0 + E)) / self.hbar

    def kappa_out(self, E):
        """
        Local wave number (real) outside potential for negative energy
        """
        return np.sqrt(-2.0 * self.mu * E) / self.hbar

    def find_E0(self):
        """
        Function to find the ground-state energy by solving a transcendental
        equation that matches inner and outer solutions.

        Uses scipy.optimize.fsolve given an initial guess.

        The guess of the energy is taken to be almost the bottom of the well.
        Should this be a fraction rather than an offset?  E.g., something like
          E_guess = 0.95 * (-self.V0)

        Would it be better to use
           from scipy.optimize import least_squares
        and minimize the sum of the squares (just the square here), because
        one can specify bounds?
        """
        from scipy.optimize import fsolve

        def f(E):
            """
            Function that is zero for E equal to a square-well bound-state energy.
            """
            k = self.k_in(E)
            kappa = self.kappa_out(E)
            return kappa / k + 1.0 / np.tan(k * self.R)

        E_guess = -self.V0 + 0.1  # The offset from -V0 is arbitrary.
        E_gs = fsolve(f, E_guess)
        return E_gs[0]

    def unnormalized_wave_function(self, r):
        ratio = np.sin(self.k_gs * self.R) / np.exp(-self.kappa_gs * self.R)
        return np.where(
            r <= self.R, np.sin(self.k_gs * r), ratio * np.exp(-self.kappa_gs * r)
        )

    def wave_function(self, r):
        return self.norm_const * self.unnormalized_wave_function(r)

    def radius_expectation(self):
        # Interior part
        k = self.k_gs
        kappa = self.kappa_gs
        R = self.R
        # interior = (
        #     2 * k * R * np.sin(k * R) + (2 - k ** 2 * R ** 2) * np.cos(k * R) - 2
        # ) / k ** 3
        # exterior = np.exp(-R * kappa) * (2 + R * kappa * (2 + R * kappa)) / kappa ** 3
        # interior = (
        #     4 * k ** 3 * R ** 3
        #     + 3 * np.sin(2 * k * R)
        #     - 6 * k * R * (np.cos(2 * k * R) + k * R * np.sin(2 * k * R))
        # ) / (24 * k ** 3)
        # exterior = np.exp(-2 * R * kappa) * (1 + 2 * R * kappa * (1 + R * kappa)) / (4 * kappa ** 3)
        kR = k * R
        interior = (1 + 2 * kR ** 2 - np.cos(2 * kR) - 2 * k * R * np.sin(2 * kR)) / (
            8 * k ** 2
        )
        exterior = np.exp(-2 * R * kappa) * (1 + 2 * R * kappa) / (4 * kappa ** 2)
        ratio = np.sin(self.k_gs * self.R) / np.exp(-self.kappa_gs * self.R)
        return self.norm_const ** 2 * (interior + ratio ** 2 * exterior)


class SquareWell:
    """
    Define a square well object with potential V(r) = -V0 \theta(R-r)

    Radial wave functions u(r) = r*R(r) are used. So far this is just l=0.

    Issues and to-do list:
     * Remove the mesh part and solve for coefficients directly
     * Need to treat the discontinuity in the potential at R explicitly,
        e.g., by splitting integrals into r <= R and r >= R.
     * Divide-by-zero run-time warnings.
     * Extend to l > 0.

    Parameters
    ----------
    V0 : float
        Depth of square well (V0 > 0).
    R : float
        Radius of square well. R = 1 by default but explicit in class.
    mu : float
        Reduced mass.  mu = 1 by default but explicit in class.
    hbar : float
        Planck's constant divided by 2\pi. hbar = 1 (and not explicit).
    r_max : float (optional, default = 20)
        Maximum radius used in defining the mesh of r.
    num_pts : int (optional, default = 2000)
        Number of r mesh points both up to R and from R to r_max.

    Methods
    -------
    Vsw(r)
        Returns the value of the potential at radius r.
    Ek(k)
        Returns the value of the kinetic energy for momentum k.
    deltaAnalytic(k)
        Returns the analytic value for the phase shift at momentum k.
    deltaAnalyticAdjusted(k)
        Returns the analytic value for the phase shift at momentum k,
         adjusted so that it is continuous.
    k_cot_delta(k)
        Returns k time the cotangent of the phase shift at momentum k.
         It will not have the possible jumps that the phase shift has.
    k_in(E)
        Returns the momentum inside the square well at energy E.
    k_out(E)
        Returns the momentum outside the square well at energy E > 0.
    kap_out(E)
        Returns the (imaginary) momentum outside the square well at energy E < 0.
    f_sw(E)
        Transcendental function that is equal to zero when the square well
         has a bound state at E.
    find_E0()
        Function to find the lowest energy bound state.
    un_wf_in(r)
        Returns the unnormalized ground state wf at r inside the well.
    un_wf_out()
        Returns unnormalized ground state wf at r outside the well.
    norm_wf_simps()
        Find the normalized ground state wf on a rMeshSW mesh.
    norm_wf_gauss()
        Find the normalized ground state wf on a rGaussSW mesh.
    """

    def __init__(
        self, V0, R=1.0, mu=1.0, hbar=1.0, r_max=20.0, num_pts=2000, gauss_pts=48
    ):
        self.V0 = V0
        self.R = R
        self.mu = mu
        self.hbar = hbar
        self.r_max = r_max  # this can be np.inf

        self.find_E0()  # find the ground-state energy
        self.find_coeffs()  # find the coefficients needed to normalize wf

        # Use a mesh with equal points from 0 to R and R to r_max
        self.num_pts = num_pts

        self.gauss_pts = int(gauss_pts)

        # self.r_mesh = rMeshSW(self.R, self.r_max, num_pts, num_pts)
        # self.gauss_mesh = rGaussSW(self.R, self.r_max, self.gauss_pts, self.gauss_pts)

        self.norm_wf_simps()  # normalized gs wave function with Simpsons rule
        self.norm_wf_gauss()  # normalized gs wave function with Gauss quadrature

    def Vsw(self, r, Vd=None):
        """
        Potential for the square well.
        Defined so that it works with scalar or numpy vector r.
        """
        if Vd is None:
            Vd = self.V0
        ans = np.where(r > self.R, 0.0, -Vd)
        return ans

    def Ek(self, k):
        """
        Kinetic energy
        """
        return (self.hbar * k) ** 2 / (2.0 * self.mu)

    def delta_analytic(self, k):
        """
        Phase shift from an analytic expression from Taylor's book.
        """
        Ekval = self.Ek(k)
        sum = Ekval + self.V0
        return np.arctan(
            np.sqrt(Ekval / sum) * np.tan(self.R * np.sqrt(2.0 * self.mu * sum))
        ) - self.R * np.sqrt(2.0 * self.mu * Ekval)

    def delta_analytic_adjusted(self, k):
        """
        Adjust phase shift so within usual range.
        Beware of bound states.
        """
        return np.arctan(np.tan(self.delta_analytic(k)))

    def k_cot_delta(self, k):
        """
        k \cot\delta(k)
        """
        return k / np.tan(self.delta_analytic(k))

    def k_in(self, E):
        """
        Local wave number inside square-well potential
        """
        return np.sqrt(2.0 * self.mu * (self.V0 + E)) / self.hbar

    def k_out(self, E):
        """
        Local wave number (real) outside potential for positive energy
        """
        return np.sqrt(2.0 * self.mu * E) / self.hbar

    def kap_out(self, E):
        """
        Local wave number (real) outside potential for negative energy
        """
        return np.sqrt(-2.0 * self.mu * E) / self.hbar

    def f_sw(self, E):
        """
        Function that is zero for E equal to a square-well bound-state energy.
        """
        k_in = self.k_in(E)
        return self.kap_out(E) / self.k_in(E) + 1.0 / np.tan(self.k_in(E) * self.R)

    def find_E0(self):
        """
        Function to find the ground-state energy by solving a transcendental
        equation that matches inner and outer solutions.

        Uses scipy.optimize.fsolve given an initial guess.

        The guess of the energy is taken to be almost the bottom of the well.
        Should this be a fraction rather than an offset?  E.g., something like
          E_guess = 0.95 * (-self.V0)

        Would it be better to use
           from scipy.optimize import least_squares
        and minimize the sum of the squares (just the square here), because
        one can specify bounds?
        """
        from scipy.optimize import fsolve

        E_guess = -self.V0 + 0.1  # The offset from -V0 is arbitrary.
        self.E_gs = fsolve(self.f_sw, E_guess)
        self.E0 = self.E_gs[0]

    def un_wf_in(self, r):
        """
        Unnormalized interior wave function at radius r.
        """
        return np.sin(self.k_in(self.E0) * r)

    def un_wf_out(self, r):
        """
        Unnormalized exterior wave function at radius r.
        """
        return np.exp(-self.kap_out(self.E0) * r)

    def un_scatt_wf_in(self, r, E):
        """
        Unnormalized interior scattering wave function with energy E at radius r.
        """
        return np.sin(self.k_in(E) * r)

    def scatt_wf_out(self, r, E):
        """
        Exterior scattering wave function with energy E at radius r, normalized
         with 1/k factor with k = np.sqrt(2*mu*E).
        """
        k = self.k_out(E)
        delta_k = self.delta_analytic_adjusted(k)
        # return np.sin(k * r + self.delta_analytic(k)) / k
        return np.sin(k * r + delta_k) / (k * np.cos(delta_k))

    def norm_wf_simps(self):
        """
        The normalized wave function, evaluated at r_pts defined by
         r_max and num_pts, given R, with num_pts from 0 to R and num_pts
         from R to r_max.
        Also find the kinetic energy operator acting on the wave function.
        """
        from scipy.integrate import simps

        r_in_pts = self.r_mesh.r_in_pts
        r_out_pts = self.r_mesh.r_out_pts

        E0_wf_in = self.un_wf_in(r_in_pts)
        norm_in = simps(E0_wf_in ** 2, r_in_pts)  # normalization for r <= R

        E0_wf_out = self.un_wf_out(r_out_pts)
        norm_out = simps(E0_wf_out ** 2, r_out_pts)  # normalization for r >= R

        ratio = E0_wf_out[0] / E0_wf_in[-1]  # at the overlapping point
        coeff_out = 1.0 / np.sqrt(norm_in * ratio ** 2 + norm_out)
        E0_wf_in = coeff_out * ratio * E0_wf_in  # rescale so normalized
        E0_wf_out = coeff_out * E0_wf_out  # rescale so normalized
        self.E0_wf_simps = np.append(E0_wf_in, E0_wf_out[1:])  # omit overlapping point

        factor = self.hbar ** 2 / (2.0 * self.mu)
        T_in = factor * self.k_in(self.E0) ** 2
        T_out = -factor * self.kap_out(self.E0) ** 2
        self.T_on_E0_wf_simps = np.append(T_in * E0_wf_in, (T_out * E0_wf_out)[1:])
        self.T_on_E0_wf_alt_simps = np.append(
            (self.E0 - self.Vsw(r_in_pts)) * E0_wf_in,
            ((self.E0 - self.Vsw(r_out_pts)) * E0_wf_out)[1:],
        )
        return self.E0_wf_simps

    def norm_wf_gauss(self):
        """
        The normalized wave function, evaluated at nodes defined by
         r_max and num_gauss, given R, with gauss_pts from 0 to R and
         gauss_pts from R to r_max.
        Also find the kinetic energy operator acting on the wave function.
        """

        r_nodes = self.gauss_mesh.nodes
        r_nodes_in = r_nodes[: self.gauss_pts]
        r_nodes_out = r_nodes[self.gauss_pts :]
        r_weights = self.gauss_mesh.weights
        r_weights_in = r_weights[: self.gauss_pts]
        r_weights_out = r_weights[self.gauss_pts :]

        # np.where(r > self.R, 0., -Vd)

        E0_wf_in = self.un_wf_in(r_nodes_in)
        norm_in = E0_wf_in ** 2 @ r_weights_in  # normalization for r <= R

        E0_wf_out = self.un_wf_out(r_nodes_out)
        norm_out = E0_wf_out ** 2 @ r_weights_out  # normalization for r >= R

        # match at the overlapping point
        ratio = self.un_wf_out(self.R) / self.un_wf_in(self.R)
        coeff_out = 1.0 / np.sqrt(norm_in * ratio ** 2 + norm_out)
        E0_wf_in = coeff_out * ratio * E0_wf_in  # rescale so normalized
        E0_wf_out = coeff_out * E0_wf_out  # rescale so normalized
        self.E0_wf_gauss = np.append(E0_wf_in, E0_wf_out)

        factor = self.hbar ** 2 / (2.0 * self.mu)
        T_in = factor * self.k_in(self.E0) ** 2
        T_out = -factor * self.kap_out(self.E0) ** 2
        self.T_on_E0_wf_gauss = np.append(T_in * E0_wf_in, T_out * E0_wf_out)
        self.T_on_E0_wf_alt_gauss = np.append(
            (self.E0 - self.Vsw(r_nodes_in)) * E0_wf_in,
            (self.E0 - self.Vsw(r_nodes_out)) * E0_wf_out,
        )
        return self.E0_wf_gauss

    def find_coeffs(self):
        """Find the coefficients for a normalized, continuous ground state wf
        """
        my_tol = 1.0e-14
        # using scipy.integrate.quadrature
        int_wf_sq_in, err = quadrature(
            lambda r: self.un_wf_in(r) ** 2, 0, self.R, tol=my_tol, rtol=my_tol
        )
        int_wf_sq_out, err = quadrature(
            lambda r: self.un_wf_out(r) ** 2,
            self.R,
            self.r_max,
            tol=my_tol,
            rtol=my_tol,
        )
        ratio = self.un_wf_out(self.R) / self.un_wf_in(self.R)
        self.coeff_out = 1.0 / np.sqrt(int_wf_sq_in * ratio ** 2 + int_wf_sq_out)
        self.coeff_in = self.coeff_out * ratio

    def norm_wf(self, r):
        """Returns the normalized wave function at r.
        """
        return np.where(
            r > self.R,
            self.coeff_out * self.un_wf_out(r),
            self.coeff_in * self.un_wf_in(r),
        )

    def scatt_ratio(self, E):
        """Returns the ratio between the normalized asymptotic wf and inner.
        """
        return self.scatt_wf_out(self.R, E) / self.un_scatt_wf_in(self.R, E)
