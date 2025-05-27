
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FlightEnv(gym.Env):
    def __init__(self, flight_plan):
        super().__init__()

        self.S = 124.65
        self.g = 9.81
        self.rho0 = 1.225
        self.CD0 = 0.025452
        self.k = 0.035815
        self.Cf1 = 0.92958
        self.Th1 = 0.95
        self.Th2 = 146590
        self.Th3 = 53872
        self.Th4 = 3.0453e-11
        self.eta_const = 0.877

        self.wind_coeffs_u = [-21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001]
        self.wind_coeffs_v = [-65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227]

        self.start, self.goal = flight_plan["start"], flight_plan["goal"]

        self.action_space = spaces.Box(
            low=np.array([0.0, -np.pi/4, -0.2], dtype=np.float32),
            high=np.array([1.0, np.pi/4, 0.2], dtype=np.float32),
            dtype=np.float32
        )

        high = np.array([1e4, 1e4, 15000, 300, 2*np.pi, 100000], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.reset()

    def wind(self, x, y, coeffs):
        lam = np.clip(x, -180, 180)
        phi = np.clip(y, -90, 90)
        terms = [1, lam, phi, lam*phi, lam**2, phi**2, lam**2*phi, lam*phi**2, lam**2*phi**2]
        return sum(c * t for c, t in zip(coeffs, terms))

    def rho(self, h):
        return self.rho0 * (1 - 2.2257e-5 * h) ** 4.2561

    def step(self, action):
        δ, μ, γ = action
        x, y, h, v, ψ, m = self.state

        rho = self.rho(h)
        CL = 2 * m * self.g / (rho * self.S * v**2 * np.cos(μ))
        CD = self.CD0 + self.k * CL**2

        ThrustMax = self.Th1 * self.Th2 * (1 - (3.28 * h) / self.Th3) + self.Th4 * (3.28 * h)**2
        η = self.eta_const * (1 + 1.943 * v / 60000)
        f = δ * ThrustMax * η * self.Cf1

        Wx = self.wind(x, y, self.wind_coeffs_u)
        Wy = self.wind(x, y, self.wind_coeffs_v)

        dx = v * np.cos(ψ) * np.cos(γ) + Wx
        dy = v * np.sin(ψ) * np.cos(γ) + Wy
        dh = v * np.sin(γ)
        dv = (δ * ThrustMax / m) - self.g * np.sin(γ) - (CD * self.S * rho * v**2) / (2 * m)
        dψ = (CL * self.S * rho * v) / (2 * m) * np.sin(μ) / np.cos(γ)
        dm = -f

        dt = 1.0

        x = np.clip(x + dx * dt, -180, 180)
        y = np.clip(y + dy * dt, -90, 90)
        h = h + dh * dt
        v = v + dv * dt
        ψ = ψ + dψ * dt
        m = m + dm * dt

        self.state = np.array([x, y, h, v, ψ, m], dtype=np.float32)

        # Reward function (final version)
        pos_error = np.linalg.norm(self.state[:2] - self.goal[:2])
        alt_error = abs(self.state[2] - self.goal[2])

        fuel_penalty = np.log(1 + f)
        reward = -0.05 - fuel_penalty - 0.001 * pos_error - 0.001 * alt_error

        reward += 0.2 * δ  # throttle teşviki

        if δ < 0.01:
            reward -= 0.5  # throttle = 0'a yakınsa sert ceza

        terminated = bool(pos_error < 1.0 and alt_error < 200)
        truncated = bool(m < 40000 or v < 50 or h < 0)

        return self.state.astype(np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        x = np.clip(self.start[0], -180, 180)
        y = np.clip(self.start[1], -90, 90)
        self.state = np.array([
            x, y, self.start[2],
            self.start[3], self.start[4], self.start[5]
        ], dtype=np.float32)
        return self.state.astype(np.float32), {}
