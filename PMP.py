import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class AircraftModel:
    def __init__(self):
        self.g = 9.81
        self.rho_0 = 1.225
        
        self.S = 124.65
        self.C_D0 = 0.025452
        self.k = 0.035815
        
        self.Cf1 = 0.92958
        self.Cf2 = 0.70057
        self.Cf3 = 1068.1
        
        self.CT1 = 0.95
        self.CT2_1 = 146590
        self.CT2_2 = 53872
        self.CT2_3 = 3.0453e-11
        
        self.wx_coef = np.array([-21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001])
        self.wy_coef = np.array([-65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227])
    
    def air_density(self, h):
        return self.rho_0 * (1 - (2.2257e-5) * h) ** 4.2586
    
    def lift_coefficient(self, m, rho, v, mu):
        return (2 * m * self.g) / (rho * self.S * v**2 * np.cos(mu))
    
    def drag_coefficient(self, C_L):
        return self.C_D0 + self.k * C_L**2
    
    def thrust_max(self, h):
        return self.CT1 * (1 - (3.28 * h) / self.CT2_2 + self.CT2_3 * (3.28 * h)**2)
    
    def eta(self, v):
        return (self.Cf1 / 60000) * (1 + (1.943 * v) / self.Cf2)
    
    def fuel_flow(self, delta, thr_max, eta):
        return delta * thr_max * eta * self.Cf1
    
    def wind_speed(self, lon, lat):
        c_x = self.wx_coef
        c_y = self.wy_coef
        
        terms = [
            1,
            lon,
            lat,
            lon * lat,
            lon**2,
            lat**2,
            lon**2 * lat,
            lon * lat**2,
            lon**2 * lat**2
        ]
        
        W_x = sum(c_x[i] * terms[i] for i in range(len(terms)))
        W_y = sum(c_y[i] * terms[i] for i in range(len(terms)))
        
        return W_x, W_y
    
    def dynamics(self, t, state, controls):
        x, y, h, v, psi, m = state
        gamma, mu, delta = controls
        
        rho = self.air_density(h)
        W_x, W_y = self.wind_speed(x, y)
        
        C_L = self.lift_coefficient(m, rho, v, mu)
        C_D = self.drag_coefficient(C_L)
        
        thr_max = self.thrust_max(h)
        thrust = delta * thr_max
        eta_val = self.eta(v)
        f = self.fuel_flow(delta, thr_max, eta_val)
        
        x_dot = v * np.cos(psi) * np.cos(gamma) + W_x
        y_dot = v * np.sin(psi) * np.cos(gamma) + W_y
        h_dot = v * np.sin(gamma)
        v_dot = (thrust / m) - self.g * np.sin(gamma) - (C_D * self.S * rho * v**2) / (2 * m)
        psi_dot = (C_L * self.S * rho * v) / (2 * m) * np.sin(mu) / np.cos(gamma)
        m_dot = -f
        
        return [x_dot, y_dot, h_dot, v_dot, psi_dot, m_dot]


class DirectCollocation:
    def __init__(self, aircraft_model, initial_state, target_state, n_nodes=20):
        self.aircraft = aircraft_model
        self.x0 = initial_state
        self.xf = target_state
        self.n_nodes = n_nodes
        
        self.gamma_bounds = (-0.2, 0.2)
        self.mu_bounds = (-np.pi/4, np.pi/4)
        self.delta_bounds = (0.1, 1.0)
        
        self.tf_guess = 3600  # seconds (1 hour)
        
    def setup_optimization(self):
        n = self.n_nodes
        
        # Decision variables per node: 6 states + 3 controls
        n_vars = n * (6 + 3) + 1  # +1 for final time
        
        # Initial guess
        x_guess = np.zeros(n_vars)
        
        # Set initial guess for states (linear interpolation from initial to target)
        for i in range(n):
            # States (linearly interpolated from x0 to xf)
            idx_start = i * 9
            alpha = i / (n - 1)
            for j in range(3):  # For x, y, h
                x_guess[idx_start + j] = (1 - alpha) * self.x0[j] + alpha * self.xf[j]
            
            # Velocity - constant
            x_guess[idx_start + 3] = self.x0[3]
            
            # Heading angle - constant
            x_guess[idx_start + 4] = self.x0[4]
            
            # Mass - linearly decreasing (assume 5% reduction)
            x_guess[idx_start + 5] = self.x0[5] * (1 - 0.05 * alpha)
            
            # Controls (reasonable initial values)
            x_guess[idx_start + 6] = 0.0  # gamma
            x_guess[idx_start + 7] = 0.0  # mu
            x_guess[idx_start + 8] = 0.5  # delta
        
        # Final time
        x_guess[-1] = self.tf_guess
        
        # Bounds for variables
        bounds = []
        
        # State bounds
        for i in range(n):
            # Position bounds (wide)
            bounds.append((0, 60))  # longitude
            bounds.append((35, 60))  # latitude
            
            # Altitude bounds
            bounds.append((5000, 12000))  # meters
            
            # Velocity bounds
            bounds.append((150, 300))  # m/s
            
            # Heading bounds
            bounds.append((-2*np.pi, 2*np.pi))  # radians
            
            # Mass bounds
            bounds.append((self.x0[5] * 0.7, self.x0[5]))  # kg
            
            # Control bounds
            bounds.append(self.gamma_bounds)  # gamma
            bounds.append(self.mu_bounds)     # mu
            bounds.append(self.delta_bounds)  # delta
        
        # Final time bounds
        bounds.append((1000, 10000))  # seconds
        
        # Define constraints
        n_constraints = 6 * (n-1) + 3  # Dynamics + terminal constraints
        
        return x_guess, bounds, n_constraints
    
    def objective_function(self, x):
        n = self.n_nodes
        tf = x[-1]
        dt = tf / (n - 1)
        
        # Calculate cost (time + fuel)
        cost = 0.05 * tf  # Time cost
        
        # Add fuel cost
        for i in range(n-1):
            idx_start = i * 9
            m_i = x[idx_start + 5]
            m_next = x[idx_start + 9 + 5]
            cost += 0.05 * (m_i - m_next)  # Fuel consumption cost
        
        return cost
    
    def constraint_function(self, x):
        n = self.n_nodes
        tf = x[-1]
        dt = tf / (n - 1)
        
        constraints = []
        
        # Extract initial state
        x0_idx = 0
        x0_extracted = x[x0_idx:x0_idx+6]
        
        # Initial state constraints
        for i in range(3):  # Only constrain x, y, h
            constraints.append(x0_extracted[i] - self.x0[i])
        
        # Dynamics constraints
        for i in range(n-1):
            idx_start = i * 9
            
            # Extract states and controls at current node
            state_i = x[idx_start:idx_start+6]
            controls_i = x[idx_start+6:idx_start+9]
            
            # Extract states at next node
            state_next = x[idx_start+9:idx_start+15]
            
            # Euler integration of dynamics
            derivatives = self.aircraft.dynamics(0, state_i, controls_i)
            state_next_euler = [state_i[j] + dt * derivatives[j] for j in range(6)]
            
            # Add constraints for each state variable
            for j in range(6):
                constraints.append(state_next[j] - state_next_euler[j])
        
        # Terminal constraints for position and altitude
        terminal_idx = (n-1) * 9
        terminal_state = x[terminal_idx:terminal_idx+6]
        
        for i in range(3):  # Only constrain x, y, h
            constraints.append(terminal_state[i] - self.xf[i])
        
        return np.array(constraints)
    
    def solve(self):
        x_guess, bounds, n_constraints = self.setup_optimization()
        
        result = minimize(
            self.objective_function,
            x_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': self.constraint_function},
            options={'maxiter': 500, 'disp': True}
        )
        
        if not result.success:
            print("Optimization failed:", result.message)
        
        # Extract results
        x_opt = result.x
        tf = x_opt[-1]
        
        # Extract states and controls
        n = self.n_nodes
        t = np.linspace(0, tf, n)
        states = np.zeros((n, 6))
        controls = np.zeros((n, 3))
        
        for i in range(n):
            idx_start = i * 9
            states[i] = x_opt[idx_start:idx_start+6]
            controls[i] = x_opt[idx_start+6:idx_start+9]
        
        return t, states, controls


def solve_flight_plan(flight_plan_number):
    aircraft = AircraftModel()
    
    flight_plans = {
        1: {
            'initial': {
                'lon': 5,
                'lat': 40,
                'h': 8000,
                'v': 210,
                'psi': 0,
                'm': 68000
            },
            'destination': {
                'lon': 32,
                'lat': 40,
                'h': 8000
            }
        },
        2: {
            'initial': {
                'lon': 30,
                'lat': 55,
                'h': 7000,
                'v': 220,
                'psi': 40 * np.pi/180,
                'm': 67000
            },
            'destination': {
                'lon': 15,
                'lat': 40,
                'h': 9000
            }
        },
        3: {
            'initial': {
                'lon': 32,
                'lat': 45,
                'h': 8000,
                'v': 210,
                'psi': 180 * np.pi/180,
                'm': 65000
            },
            'destination': {
                'lon': 5,
                'lat': 45,
                'h': 7000
            }
        }
    }
    
    plan = flight_plans[flight_plan_number]
    
    initial_state = [
        plan['initial']['lon'],
        plan['initial']['lat'],
        plan['initial']['h'],
        plan['initial']['v'],
        plan['initial']['psi'],
        plan['initial']['m']
    ]
    
    target_state = [
        plan['destination']['lon'],
        plan['destination']['lat'],
        plan['destination']['h'],
        0, 0, 0
    ]
    
    optimizer = DirectCollocation(aircraft, initial_state, target_state, n_nodes=30)
    
    t, states, controls = optimizer.solve()
    
    flight_time = t[-1]
    initial_mass = states[0, 5]
    final_mass = states[-1, 5]
    fuel_consumption = initial_mass - final_mass
    
    print(f"Flight {flight_plan_number} Results:")
    print(f"Flight time: {flight_time/60:.2f} minutes")
    print(f"Fuel consumption: {fuel_consumption:.2f} kg")
    
    plot_results(t, states, controls, flight_plan_number)
    
    return t, states, controls, flight_time, fuel_consumption


def plot_results(t, states, controls, flight_plan_number):
    t_min = t / 60
    
    x = states[:, 0]
    y = states[:, 1]
    h = states[:, 2]
    v = states[:, 3]
    psi = states[:, 4]
    m = states[:, 5]
    
    gamma = controls[:, 0]
    mu = controls[:, 1]
    delta = controls[:, 2]
    
    aircraft = AircraftModel()
    thrust = np.zeros_like(t)
    for i in range(len(t)):
        thr_max = aircraft.thrust_max(h[i])
        thrust[i] = delta[i] * thr_max
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 3, 1)
    plt.plot(x, y)
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.title('x-y Trajectory')
    plt.grid(True)
    
    plt.subplot(3, 3, 2)
    plt.plot(t_min, h)
    plt.xlabel('Time [min]')
    plt.ylabel('Altitude [m]')
    plt.title('Altitude vs Time')
    plt.grid(True)
    
    plt.subplot(3, 3, 3)
    plt.plot(t_min, v)
    plt.xlabel('Time [min]')
    plt.ylabel('True Airspeed [m/s]')
    plt.title('Velocity vs Time')
    plt.grid(True)
    
    plt.subplot(3, 3, 4)
    plt.plot(t_min, m)
    plt.xlabel('Time [min]')
    plt.ylabel('Aircraft Mass [kg]')
    plt.title('Mass vs Time')
    plt.grid(True)
    
    plt.subplot(3, 3, 5)
    plt.plot(t_min, thrust)
    plt.xlabel('Time [min]')
    plt.ylabel('Thrust [N]')
    plt.title('Thrust vs Time')
    plt.grid(True)
    
    plt.subplot(3, 3, 6)
    plt.plot(t_min, delta)
    plt.xlabel('Time [min]')
    plt.ylabel('Throttle [-]')
    plt.title('Throttle vs Time')
    plt.grid(True)
    
    plt.subplot(3, 3, 7)
    plt.plot(t_min, np.degrees(mu))
    plt.xlabel('Time [min]')
    plt.ylabel('Bank Angle [deg]')
    plt.title('Bank Angle vs Time')
    plt.grid(True)
    
    plt.subplot(3, 3, 8)
    plt.plot(t_min, np.degrees(gamma))
    plt.xlabel('Time [min]')
    plt.ylabel('Flight Path Angle [deg]')
    plt.title('Flight Path Angle vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f'Flight Plan {flight_plan_number} Results', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    plt.savefig(f'flight_plan_{flight_plan_number}_results.png')
    plt.close()


def main():
    results = {}
    
    for flight_plan in [1, 2, 3]:
        print(f"\nSolving Flight Plan {flight_plan}...")
        t, states, controls, flight_time, fuel_consumption = solve_flight_plan(flight_plan)
        
        results[flight_plan] = {
            'flight_time': flight_time,
            'fuel_consumption': fuel_consumption
        }
    
    print("\nSummary of Results:")
    print("------------------")
    print("Flight Plan | Flight Time (min) | Fuel Consumption (kg)")
    print("--------------------------------------------------")
    for flight_plan, result in results.items():
        print(f"{flight_plan:11d} | {result['flight_time']/60:15.2f} | {result['fuel_consumption']:20.2f}")


if __name__ == "__main__":
    main()