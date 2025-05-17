import numpy as np

# --- Constants for B737-800 ---
C_FC = [0.92958, 0.70057, 1068.1]        # C_FC0, C_FC1, C_FC2
THR = [0.95, 146590, 53872, 3.0453e-11]  # C_THR0,1,2,3
AERO = [0.025452, 0.035815]              # S, k
S = 124.65
g = 9.80665

# --- Wind Field Polynomial Coefficients ---
wind_coeffs_x = [-21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001]
wind_coeffs_y = [-65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227]

def wind_field(x, y, coeffs):
    # x = longitude (lambda), y = latitude (phi)
    terms = [
        1, x, y, x*y, x**2, y**2,
        x**2 * y, x * y**2, x**2 * y**2
    ]
    return sum(c*t for c, t in zip(coeffs, terms))

def compute_rho(h):
    return 1.225 * (1 - 2.2257e-5 * h) ** 4.2561

def compute_thrust_max(h):
    C0, C1, C2, C3 = THR
    return C0 * C1 * (1 - 3.28 * h / C2) + C3 * (3.28 * h)**2

def compute_eta(v):
    return C_FC[2] / 60000 * (1 + 1.943 * v / C_FC[2])

def compute_lift_coeff(m, rho, v, mu):
    return 2 * m * g / (rho * S * v**2 * np.cos(mu))

def compute_drag_coeff(CL):
    return AERO[0] + AERO[1] * CL**2

def compute_fuel_flow(delta, T_max, eta):
    return delta * T_max * eta * C_FC[0]

def aircraft_dynamics(t, state, control):
    # Unpack state
    x, y, h, v, psi, m = state
    gamma, mu, delta = control  # control inputs

    # Compute wind components
    Wx = wind_field(x, y, wind_coeffs_x)
    Wy = wind_field(x, y, wind_coeffs_y)

    # Atmospheric and aerodynamic quantities
    rho = compute_rho(h)
    T_max = compute_thrust_max(h)
    eta = compute_eta(v)
    CL = compute_lift_coeff(m, rho, v, mu)
    CD = compute_drag_coeff(CL)
    f = compute_fuel_flow(delta, T_max, eta)

    # State derivatives
    dx = v * np.cos(psi) * np.cos(gamma) + Wx
    dy = v * np.sin(psi) * np.cos(gamma) + Wy
    dh = v * np.sin(gamma)
    dv = (delta * T_max / m) - g * np.sin(gamma) - (CD * S * rho * v**2) / (2 * m)
    dpsi = (CL * S * rho * v) / (2 * m * np.cos(gamma)) * np.sin(mu)
    dm = -f

    return [dx, dy, dh, dv, dpsi, dm]
