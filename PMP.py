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
    def __init__(self, aircraft_model, initial_state, target_state, n_nodes=30):
        self.aircraft = aircraft_model
        self.x0 = initial_state
        self.xf = target_state
        self.n_nodes = n_nodes

        # Adjust bounds for better convergence
        self.gamma_bounds = (-0.15, 0.15)  # Slightly narrower
        self.mu_bounds = (-np.pi/6, np.pi/6)  # Less aggressive bank
        self.delta_bounds = (0.2, 0.9)     # More realistic throttle range

        # Estimate flight time based on distance
        dx = self.xf[0] - self.x0[0]
        dy = self.xf[1] - self.x0[1]
        dist = np.sqrt(dx**2 + dy**2) * 111  # Approx km (1 degree ≈ 111 km)
        
        # Estimate time in seconds (assuming ~800 km/h cruise speed)
        self.tf_guess = dist * 4.5  # More realistic estimate
        
    def setup_optimization(self):
        n = self.n_nodes
        n_vars = n * (6 + 3) + 1  # states + controls + tf
        x_guess = np.zeros(n_vars)

        # Make better initial guesses
        for i in range(n):
            idx_start = i * 9
            alpha = i / (n - 1)
            
            # State variables
            # Linear interpolation for position and altitude
            for j in range(3):
                x_guess[idx_start + j] = (1 - alpha) * self.x0[j] + alpha * self.xf[j]
            
            # Speed varies smoothly (slight decrease during climb, etc.)
            x_guess[idx_start + 3] = self.x0[3] * (1 - 0.05 * np.sin(np.pi * alpha))
            
            # Heading turns gradually toward destination
            target_heading = np.arctan2(self.xf[1] - self.x0[1], self.xf[0] - self.x0[0])
            if target_heading < 0:
                target_heading += 2 * np.pi
            
            initial_heading = self.x0[4]
            diff = target_heading - initial_heading
            # Ensure we take the shorter path
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
                
            x_guess[idx_start + 4] = initial_heading + alpha * diff
            
            # Mass decreases gradually
            x_guess[idx_start + 5] = self.x0[5] * (1 - 0.05 * alpha)
            
            # Control variables with smoother profile
            # Flight path angle - slight climb then descent
            x_guess[idx_start + 6] = 0.05 * np.sin(np.pi * alpha)
            
            # Bank angle - varies to turn toward destination
            if i < n/3 or i > 2*n/3:
                # Bank more at beginning and end for turns
                turn_factor = 0.2 * np.sin(3 * np.pi * alpha)
                if diff < 0:
                    turn_factor *= -1
                x_guess[idx_start + 7] = turn_factor
            else:
                x_guess[idx_start + 7] = 0.0
            
            # Throttle - higher during climb, lower during cruise
            if i < n/5:
                x_guess[idx_start + 8] = 0.85  # Initial climb
            elif i > 4*n/5:
                x_guess[idx_start + 8] = 0.6   # Final approach
            else:
                x_guess[idx_start + 8] = 0.7   # Cruise

        # Final time estimate
        x_guess[-1] = self.tf_guess

        # Define bounds with more realistic constraints
        bounds = []
        for i in range(n):
            # State bounds
            bounds.append((0, 60))  # Longitude
            bounds.append((35, 60))  # Latitude
            bounds.append((5000, 12000))  # Altitude
            bounds.append((150, 270))  # Speed (m/s) - more realistic max
            bounds.append((-2*np.pi, 2*np.pi))  # Heading
            bounds.append((self.x0[5] * 0.8, self.x0[5]))  # Mass
            
            # Control bounds
            bounds.append(self.gamma_bounds)  # Flight path angle
            bounds.append(self.mu_bounds)  # Bank angle
            bounds.append(self.delta_bounds)  # Throttle

        # Flight time bounds
        min_time = self.tf_guess * 0.7  # Allow some flexibility
        max_time = self.tf_guess * 1.5
        bounds.append((min_time, max_time))

        # Number of constraints: continuity + boundary
        n_constraints = 6 * (n-1) + 6
        return x_guess, bounds, n_constraints
    
    def objective_function(self, x):
        n = self.n_nodes
        tf = x[-1]
        dt = tf / (n - 1)

        # Initialize cost components
        fuel_cost_total = 0.0
        time_cost = 0.05 * tf  # Time cost component (scaled)
        control_smoothness_cost = 0.0
        terminal_cost = 0.0
        
        # Calculate costs
        for i in range(n-1):
            idx = i * 9
            
            # Extract states and controls
            m_i = x[idx + 5]
            m_next = x[idx + 9 + 5]
            gamma_i = x[idx + 6]
            mu_i = x[idx + 7]
            delta_i = x[idx + 8]
            v_i = x[idx + 3]
            
            # Fuel consumption cost
            fuel_cost = m_i - m_next
            fuel_cost_total += fuel_cost
            
            # Penalize extreme throttle settings
            if delta_i > 0.85:
                control_smoothness_cost += 50.0 * (delta_i - 0.85)**2
            elif delta_i < 0.3:
                control_smoothness_cost += 50.0 * (0.3 - delta_i)**2
            
            # Penalize extreme flight path angles
            if abs(gamma_i) > 0.1:
                control_smoothness_cost += 100.0 * (abs(gamma_i) - 0.1)**2
            
            # Penalize extreme bank angles
            if abs(mu_i) > np.pi/8:
                control_smoothness_cost += 50.0 * (abs(mu_i) - np.pi/8)**2
            
            # Speed constraint - prefer efficient cruise speed
            if abs(v_i - 220) > 30:
                control_smoothness_cost += 5.0 * (abs(v_i - 220) - 30)**2

        # Control rate smoothness
        for i in range(n-2):
            idx_i = i * 9
            idx_ip1 = (i + 1) * 9
            
            dgamma = x[idx_ip1 + 6] - x[idx_i + 6]
            dmu = x[idx_ip1 + 7] - x[idx_i + 7]
            ddelta = x[idx_ip1 + 8] - x[idx_i + 8]
            
            # Penalize rapid control changes
            control_smoothness_cost += 20.0 * dt * (
                (dgamma/dt)**2 + 
                (dmu/dt)**2 + 
                (ddelta/dt)**2
            )
        
        # Terminal conditions cost
        end_idx = (n-1) * 9
        x_end = x[end_idx]
        y_end = x[end_idx + 1]
        h_end = x[end_idx + 2]
        
        terminal_cost = 1000.0 * (
            (x_end - self.xf[0])**2 + 
            (y_end - self.xf[1])**2 + 
            (h_end - self.xf[2])**2 / 1000000  # Scale height error
        )
        
        # Combine all cost components
        total_cost = fuel_cost_total + time_cost + control_smoothness_cost + terminal_cost
        
        return total_cost
    
    def constraint_function(self, x):
        n = self.n_nodes
        tf = x[-1]
        dt = tf / (n - 1)
        
        constraints = []
        
        # Initial state constraints (all 6 states)
        init_idx = 0
        for i in range(6):
            constraints.append(x[init_idx + i] - self.x0[i])
        
        # Dynamics constraints using trapezoidal integration (more accurate than Euler)
        for i in range(n-1):
            idx_i = i * 9
            idx_ip1 = idx_i + 9
            
            # Extract states and controls at node i
            state_i = x[idx_i:idx_i+6]
            controls_i = x[idx_i+6:idx_i+9]
            
            # Extract states at node i+1
            state_ip1 = x[idx_ip1:idx_ip1+6]
            controls_ip1 = x[idx_ip1+6:idx_ip1+9]
            
            # Compute derivatives at node i
            deriv_i = self.aircraft.dynamics(0, state_i, controls_i)
            
            # Compute derivatives at node i+1
            deriv_ip1 = self.aircraft.dynamics(0, state_ip1, controls_ip1)
            
            # Trapezoidal integration
            for j in range(6):
                predicted = state_i[j] + 0.5 * dt * (deriv_i[j] + deriv_ip1[j])
                constraints.append(state_ip1[j] - predicted)
        
        # Terminal constraints for position and altitude
        terminal_idx = (n-1) * 9
        for i in range(3):  # Only constrain x, y, h
            constraints.append(x[terminal_idx + i] - self.xf[i])
        
        return np.array(constraints)
    
    def solve(self):
        x_guess, bounds, n_constraints = self.setup_optimization()
        
        # First attempt with looser tolerances to get closer to solution
        result = minimize(
            self.objective_function,
            x_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': self.constraint_function},
            options={
                'maxiter': 300,
                'ftol': 1e-6,
                'eps': 1e-3,
                'disp': True
            }
        )
        
        # Refine solution with tighter tolerances
        if result.success:
            print("Initial optimization successful, refining solution...")
            result = minimize(
                self.objective_function,
                result.x,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': self.constraint_function},
                options={
                    'maxiter': 200,
                    'ftol': 1e-8,
                    'eps': 1e-5,
                    'disp': True
                }
            )
        else:
            print("Initial optimization did not fully converge:", result.message)
            print("Attempting to extract usable solution anyway...")
        
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
        
        # Check final position error
        final_pos_error = np.sqrt((states[-1, 0] - self.xf[0])**2 + 
                                 (states[-1, 1] - self.xf[1])**2)
        print(f"Final position error: {final_pos_error:.4f} degrees")
        
        # Check altitude error
        alt_error = abs(states[-1, 2] - self.xf[2])
        print(f"Final altitude error: {alt_error:.2f} meters")
        
        return t, states, controls, result.success


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
        0, 0, 0  # We don't constrain final velocity, heading or mass
    ]
    
    # Estimate distance for each flight plan
    dx = target_state[0] - initial_state[0]
    dy = target_state[1] - initial_state[1]
    dist_km = np.sqrt(dx**2 + dy**2) * 111  # Approx km (1 degree ≈ 111 km)
    print(f"Flight {flight_plan_number} - Approximate distance: {dist_km:.1f} km")
    
    # More nodes for longer flights
    if dist_km > 2000:
        n_nodes = 40
    else:
        n_nodes = 30
    
    optimizer = DirectCollocation(aircraft, initial_state, target_state, n_nodes=n_nodes)
    
    t, states, controls, success = optimizer.solve()
    
    flight_time = t[-1]
    initial_mass = states[0, 5]
    final_mass = states[-1, 5]
    fuel_consumption = initial_mass - final_mass
    
    print(f"Flight {flight_plan_number} Results:")
    print(f"Flight time: {flight_time/60:.2f} minutes")
    print(f"Fuel consumption: {fuel_consumption:.2f} kg")
    print(f"Average speed: {dist_km/(flight_time/3600):.1f} km/h")
    
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
    
    # Calculate fuel flow over time
    fuel_flow = np.zeros_like(t)
    for i in range(len(t)):
        thr_max = aircraft.thrust_max(h[i])
        eta_val = aircraft.eta(v[i])
        fuel_flow[i] = aircraft.fuel_flow(delta[i], thr_max, eta_val)
    
    # Calculate wind vectors for plotting
    wind_x = np.zeros_like(t)
    wind_y = np.zeros_like(t)
    for i in range(len(t)):
        wind_x[i], wind_y[i] = aircraft.wind_speed(x[i], y[i])
    
    plt.figure(figsize=(15, 12))
    
    # Trajectory plot with wind field arrows
    plt.subplot(3, 3, 1)
    plt.plot(x, y, 'b-', linewidth=2)
    plt.plot(x[0], y[0], 'go', markersize=8, label='Start')
    plt.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
    
    # Add wind vectors (subsample for clarity)
    arrow_indices = np.linspace(0, len(t)-1, 10).astype(int)
    for i in arrow_indices:
        plt.arrow(x[i], y[i], wind_x[i]/20, wind_y[i]/20, 
                 head_width=0.2, head_length=0.3, fc='r', ec='r', alpha=0.5)
    
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.title('x-y Trajectory with Wind')
    plt.grid(True)
    plt.legend()
    
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
    
    plt.subplot(3, 3, 9)
    plt.plot(t_min, fuel_flow)
    plt.xlabel('Time [min]')
    plt.ylabel('Fuel Flow [kg/s]')
    plt.title('Fuel Flow vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f'Flight Plan {flight_plan_number} Results', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    plt.savefig(f'flight_plan_{flight_plan_number}_results.png')
    plt.close()
    
    # Additional plot for wind field visualization
    plt.figure(figsize=(10, 8))
    
    # Create a grid for wind field visualization
    lon_grid = np.linspace(min(x)-5, max(x)+5, 20)
    lat_grid = np.linspace(min(y)-5, max(y)+5, 20)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    
    # Calculate wind at each grid point
    U = np.zeros_like(LON)
    V = np.zeros_like(LAT)
    for i in range(LON.shape[0]):
        for j in range(LON.shape[1]):
            U[i,j], V[i,j] = aircraft.wind_speed(LON[i,j], LAT[i,j])
    
    # Wind speed magnitude
    wind_speed = np.sqrt(U**2 + V**2)
    
    # Plot wind field
    plt.contourf(LON, LAT, wind_speed, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Wind Speed [m/s]')
    
    # Plot trajectory
    plt.plot(x, y, 'r-', linewidth=2)
    plt.plot(x[0], y[0], 'go', markersize=8, label='Start')
    plt.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
    
    # Plot wind vectors
    plt.quiver(LON[::2, ::2], LAT[::2, ::2], U[::2, ::2], V[::2, ::2], 
              scale=200, color='black', alpha=0.7)
    
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.title(f'Flight Plan {flight_plan_number} - Trajectory and Wind Field')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(f'flight_plan_{flight_plan_number}_wind_field.png')
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