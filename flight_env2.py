import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FlightEnv(gym.Env):
    def __init__(self, flight_plan):
        super().__init__()

        # Aircraft parameters (B737-800)
        self.S = 124.65  # Wing area (m^2)
        self.g = 9.81    # Gravity (m/s^2)
        self.rho0 = 1.225  # Sea level air density (kg/m^3)
        
        # Aerodynamic coefficients
        self.CD0 = 0.025452
        self.k = 0.035815
        
        # Fuel consumption coefficients
        self.Cf1 = 0.92958
        self.Cf2 = 0.70057
        self.Cf3 = 1068.1
        
        # Thrust coefficients
        self.CTc1 = 0.95
        self.CTc2 = 146590
        self.CTc3 = 53872
        self.CTc4 = 3.0453e-11
        
        # Efficiency coefficient
        self.Cf1_const = 0.877

        # Wind coefficients
        self.wind_coeffs_u = [-21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001]
        self.wind_coeffs_v = [-65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227]

        # Flight plan
        self.start, self.goal = flight_plan["start"], flight_plan["goal"]
        
        # Time step
        self.dt = 1.0  # seconds
        
        # Episode tracking
        self.max_episode_steps = 10000  # Maximum steps per episode
        self.current_step = 0

        # Action space: [throttle δ, bank angle μ, flight path angle γ]
        self.action_space = spaces.Box(
            low=np.array([0.0, -np.pi/4, -0.1], dtype=np.float32),
            high=np.array([1.0, np.pi/4, 0.1], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: [x, y, h, v, ψ, m]
        high = np.array([180, 90, 15000, 300, 2*np.pi, 100000], dtype=np.float32)
        low = np.array([-180, -90, 0, 50, 0, 1000], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def wind(self, x, y, coeffs):
        """Calculate wind speed at given position"""
        lam = np.clip(x, -180, 180)
        phi = np.clip(y, -90, 90)
        terms = [1, lam, phi, lam*phi, lam**2, phi**2, lam**2*phi, lam*phi**2, lam**2*phi**2]
        return sum(c * t for c, t in zip(coeffs, terms))

    def rho(self, h):
        """Air density at altitude h (meters)"""
        return self.rho0 * (1 - 2.2257e-5 * h) ** 4.2561

    def step(self, action):
        # Extract action components
        δ = np.clip(action[0], 0.0, 1.0)  # Throttle
        μ = np.clip(action[1], -np.pi/4, np.pi/4)  # Bank angle
        γ = np.clip(action[2], -0.1, 0.1)  # Flight path angle
        
        # Current state
        x, y, h, v, ψ, m = self.state
        
        # Air density at current altitude
        rho = self.rho(h)
        
        # Lift coefficient (from equilibrium in cruise)
        CL = 2 * m * self.g / (rho * self.S * v**2 * np.cos(μ))
        
        # Drag coefficient
        CD = self.CD0 + self.k * CL**2
        
        # Maximum thrust available
        ThrustMax = self.CTc1 * self.CTc2 * (1 - (3.28 * h) / self.CTc3) + self.CTc4 * (3.28 * h)**2
        
        # Efficiency factor
        η = self.Cf1_const * (1 + 1.943 * v / 60000)
        
        # Fuel flow (kg/s) - Corrected formula
        f = (δ * ThrustMax * self.Cf1) / (1000 * v)  # Divided by 1000*v for realistic values
        
        # Wind components
        Wx = self.wind(x, y, self.wind_coeffs_u)
        Wy = self.wind(x, y, self.wind_coeffs_v)
        
        # State derivatives
        dx = v * np.cos(ψ) * np.cos(γ) + Wx
        dy = v * np.sin(ψ) * np.cos(γ) + Wy
        dh = v * np.sin(γ)
        dv = (δ * ThrustMax / m) - self.g * np.sin(γ) - (CD * self.S * rho * v**2) / (2 * m)
        dψ = (CL * self.S * rho * v) / (2 * m) * np.sin(μ) / np.cos(γ)
        dm = -f
        
        # Update state with time step
        x_new = x + dx * self.dt
        y_new = y + dy * self.dt
        h_new = h + dh * self.dt
        v_new = v + dv * self.dt
        ψ_new = ψ + dψ * self.dt
        m_new = m + dm * self.dt
        
        # Ensure state remains within bounds
        x_new = np.clip(x_new, -180, 180)
        y_new = np.clip(y_new, -90, 90)
        h_new = np.clip(h_new, 100, 15000)  # Minimum altitude 100m
        v_new = np.clip(v_new, 50, 300)  # Reasonable velocity bounds
        ψ_new = ψ_new % (2 * np.pi)  # Wrap heading angle
        m_new = max(m_new, 1000)  # Minimum mass
        
        self.state = np.array([x_new, y_new, h_new, v_new, ψ_new, m_new], dtype=np.float32)
        
        # Calculate reward
        reward = self._calculate_reward(f, δ, γ, μ)
        
        # Check termination conditions
        pos_error = np.sqrt((x_new - self.goal[0])**2 + (y_new - self.goal[1])**2)
        alt_error = abs(h_new - self.goal[2])
        
        # Success: reached destination
        terminated = bool(pos_error < 0.5 and alt_error < 100)
        
        # Failure conditions
        self.current_step += 1
        truncated = bool(
            m_new < 20000 or  # Too much fuel consumed
            v_new < 70 or     # Too slow
            h_new < 200 or    # Too low
            self.current_step >= self.max_episode_steps  # Episode too long
        )
        
        # Bonus for success
        if terminated:
            reward += 1000
        
        return self.state.astype(np.float32), reward, terminated, truncated, {
            'fuel_flow': f,
            'position_error': pos_error,
            'altitude_error': alt_error
        }
    
    def _calculate_reward(self, fuel_flow, throttle, gamma, mu):
        """Calculate step reward"""
        x, y, h, v, ψ, m = self.state
        
        # Distance to goal
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        
        # Altitude error
        altitude_error = abs(h - self.goal[2])
        
        # Heading towards goal
        desired_heading = np.arctan2(dy, dx)
        heading_error = abs(ψ - desired_heading)
        heading_error = min(heading_error, 2*np.pi - heading_error)  # Wrap around
        
        # Reward components
        reward = 0.0
        
        # Progress reward (positive for moving towards goal)
        prev_distance = getattr(self, 'prev_distance', distance_to_goal)
        progress = prev_distance - distance_to_goal
        reward += 10 * progress  # Reward for getting closer
        self.prev_distance = distance_to_goal
        
        # Penalty for distance from goal
        reward -= 0.01 * distance_to_goal
        
        # Penalty for altitude error
        reward -= 0.001 * altitude_error
        
        # Penalty for heading error
        reward -= 0.1 * heading_error
        
        # Fuel efficiency penalty (realistic range)
        reward -= 0.001 * fuel_flow
        
        # Encourage cruise flight (small gamma, small mu)
        reward -= 0.1 * abs(gamma)
        reward -= 0.05 * abs(mu)
        
        # Penalty for extreme throttle settings
        if throttle < 0.3 or throttle > 0.9:
            reward -= 0.1
        
        return reward
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state to initial conditions
        x0 = np.clip(self.start[0], -180, 180)
        y0 = np.clip(self.start[1], -90, 90)
        h0 = self.start[2]
        v0 = self.start[3]
        ψ0 = self.start[4]
        m0 = self.start[5]
        
        self.state = np.array([x0, y0, h0, v0, ψ0, m0], dtype=np.float32)
        
        # Reset tracking variables
        self.current_step = 0
        self.prev_distance = np.sqrt((x0 - self.goal[0])**2 + (y0 - self.goal[1])**2)
        
        return self.state.astype(np.float32), {}