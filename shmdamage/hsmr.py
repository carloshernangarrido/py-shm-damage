import numpy as np
from scipy.optimize import minimize


def fun(x, c, m_, norm):
    dyn_stiff = x[0]
    s_0 = x[1]
    m_0 = x[2]
    x_locs = np.arange(len(m_))
    __M = m_ + m_0 + x_locs * s_0
    error = c - __M / dyn_stiff
    loss = np.linalg.norm(error, ord=norm)
    return loss


def synthesis(c: np.ndarray, q: np.ndarray, dyn_stiff_initial_guess: float = 1,
              s_0_initial_guess: float = 0, m_0_initial_guess: float = 0, norm: int = 1,
              dyn_stiff_min: float = -np.inf, dyn_stiff_max: float = np.inf,
              s_0_min: float = -np.inf, s_0_max: float = np.inf,
              m_0_min: float = -np.inf, m_0_max: float = np.inf):
    """
    A function that returns a curvature corresponding to the healthy structure
    (beam) in a Region Of Interest (ROI), given the curvature of the actual
    structure (beam) c and the load q applied to the corresponding ROI. This
    function assumes the damage in the ROI is sparse.
    References: Garrido, Domizio, Curadelli, Ambrosini. Synthesis of
    healthy-structure model responses for damage quantification, Structural Health
    Monitoring (Internation Journal), 2022.

    Parameters
    ----------
    c : ndarray(n,)
        Curvature shape of the actual beam in the ROI.
    q : ndarray(n,)
        Loading shape of the actual beam in the ROI.
    dyn_stiff_initial_guess : float
        Initial guess for finding the dynamic stiffness of the beam on the ROI. By default, = 1.
    m_0_initial_guess : float
        Initial guess for finding the bending moment at the initial end of the ROI of the beam on the ROI. By default,
        = 0.
    s_0_initial_guess : float
        Initial guess for finding the shear at the initial end of the ROI of the beam on the ROI. By default, = 0.
    norm : int
        Norm used for the fitting of the healthy c_h curvature to the (possibly) damaged curvature c.
    dyn_stiff_min : float
        Minimum allowable dynamic stiffness in the fit searching. By default, = -np.inf.
    dyn_stiff_max : float
        Minimum allowable dynamic stiffness in the fit searching. By default, = np.inf.
    s_0_min : float
        Minimum allowable shear at the initial end of the ROI in the fit searching. By default, = -np.inf.
    s_0_max : float
        Minimum allowable shear at the initial end of the ROI in the fit searching. By default, = np.inf.
    m_0_min : float
        Minimum allowable bending moment at the initial end of the ROI in the fit searching. By default, = -np.inf.
    m_0_max : float
        Minimum allowable bending moment at the initial end of the ROI in the fit searching. By default, = np.inf.

    Returns
    -------
    c_h : ndarray(n,)
        Curvature shape of the healthy ROI.
    m_ : ndarray(n,)
        Bending moment with null integration constants.
    m : ndarray(n,)
        Bending moment with appropriate integration constants and dynamic stiffness.
    res_x_dict : dict(3,)
        Dynamic stiffness, initial shear and initial bending moment.
    """
    bounds = ((dyn_stiff_min, dyn_stiff_max), (s_0_min, s_0_max), (m_0_min, m_0_max))
    # First approach using p-norm = 2
    x0 = np.array([dyn_stiff_initial_guess, s_0_initial_guess, m_0_initial_guess])  # dyn_stiff, s_0, m_0
    # bending moment with null integration constants
    m_ = np.cumsum(np.cumsum(q))
    # normalization
    m_ = m_ * np.linalg.norm(c, ord=np.inf) / np.linalg.norm(m_, ord=np.inf)
    # bending moment with appropriate integration constants and dynamic stiffness
    res = minimize(fun, x0, method='SLSQP', args=(c, m_, 2), bounds=bounds)
    dyn_stiff = res.x[0]
    s_0 = res.x[1]  # initial shear
    m_0 = res.x[2]  # initial bending moment

    # Final approach using p-norm = norm
    x0 = np.array([dyn_stiff, s_0, m_0])
    # bending moment with null integration constants
    m_ = np.cumsum(np.cumsum(q))
    # normalization
    m_ = m_ * np.linalg.norm(c, ord=np.inf) / np.linalg.norm(m_, ord=np.inf)
    # bending moment with appropriate integration constants and dynamic stiffness
    res = minimize(fun, x0, method='SLSQP', args=(c, m_, norm))
    dyn_stiff = res.x[0]  # dynamic stiffness
    s_0 = res.x[1]  # initial shear
    m_0 = res.x[2]  # initial bending moment

    res_x = (dyn_stiff, s_0, m_0)
    m = m_ + m_0 + np.arange(len(m_)) * s_0
    # Healthy curvature
    c_h = m / dyn_stiff
    res_x_dict = {'dyn_stiff': res_x[0],
                  's_0': res_x[1],
                  'm_0': res_x[2]}
    return c_h, m_, m, res_x_dict
