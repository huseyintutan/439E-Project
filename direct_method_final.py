import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import json

class AircraftModel:
    def __init__(self):
        self.g = 9.81
        self.rho_0 = 1.225
        self.S = 124.65
        self.C_D0 = 0.025452
        self.k = 0.035815
        
        # Fuel consumption coefficients
        self.Cf1 = 0.92958
        self.Cf2 = 0.70057
        self.Cf3 = 1068.1
        
        # Thrust coefficients
        self.CT1 = 0.95
        self.CT2_1 = 146590
        self.CT2_2 = 53872
        self.CT2_3 = 3.0453e-11
        
        # Wind coefficients
        self.wx_coef = np.array([-21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001])
        self.wy_coef = np.array([-65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227])

    def air_density(self, h):
        return self.rho_0 * (1 - 2.2257e-5 * h) ** 4.2561

    def thrust_max(self, h):
        return self.CT1 * self.CT2_1 * (1 - (3.28 * h) / self.CT2_2) + self.CT2_3 * (3.28 * h)**2

    def eta(self, v):
        return (self.Cf3 / 60000) * (1 + (1.943 * v) / self.Cf3)

    def fuel_flow(self, delta, thr_max, eta):
        flow = delta * thr_max * eta * self.Cf1 / 1e6
        return max(0.001, flow)

    def wind_speed(self, lon, lat):
        terms = [1, lon, lat, lon*lat, lon**2, lat**2, 
                lon**2*lat, lon*lat**2, lon**2*lat**2]
        
        W_x = sum(c*t for c, t in zip(self.wx_coef, terms))
        W_y = sum(c*t for c, t in zip(self.wy_coef, terms))
        
        return W_x, W_y

    def lift_coefficient(self, m, rho, v, mu):
        return (2 * m * self.g) / (rho * self.S * v**2 * np.cos(mu))

    def drag_coefficient(self, CL):  
        return self.C_D0 + self.k * CL**2

    def dynamics(self, t, state, controls):
        x, y, h, v, psi, m = state
        gamma, mu, delta = controls

        # Clamp controls
        gamma = np.clip(gamma, -0.1, 0.1)
        mu = np.clip(mu, -np.pi/6, np.pi/6)
        delta = np.clip(delta, 0.3, 0.9)

        # Environmental conditions
        rho = self.air_density(h)
        W_x, W_y = self.wind_speed(x, y)

        # Aerodynamic coefficients
        CL = self.lift_coefficient(m, rho, v, mu)
        CD = self.drag_coefficient(CL)

        # Propulsion
        thr_max = self.thrust_max(h)
        thrust = delta * thr_max
        eta_val = self.eta(v)
        f = self.fuel_flow(delta, thr_max, eta_val)

        # State derivatives
        x_dot = v * np.cos(psi) * np.cos(gamma) + W_x
        y_dot = v * np.sin(psi) * np.cos(gamma) + W_y
        h_dot = v * np.sin(gamma)
        v_dot = (thrust / m) - self.g * np.sin(gamma) - (CD * self.S * rho * v**2) / (2 * m)
        psi_dot = (CL * self.S * rho * v) / (2 * m) * np.sin(mu) / np.cos(gamma)
        m_dot = -f

        return [x_dot, y_dot, h_dot, v_dot, psi_dot, m_dot]


class FastDirectMethod:
    def __init__(self, aircraft_model, initial_state, target_state):
        self.aircraft = aircraft_model
        self.x0 = initial_state
        self.xf = target_state
        
        # MUCH simpler - only 15 nodes for fast convergence
        self.n_nodes = 15
        self.n_segments = self.n_nodes - 1
        
        # Calculate distance
        dx = self.xf[0] - self.x0[0]
        dy = self.xf[1] - self.x0[1]
        self.distance_km = np.sqrt(dx**2 + dy**2) * 111
        
        # Quick time estimate
        self.tf_estimate = (self.distance_km / 800) * 3600  # 800 km/h
        
        print(f"⚡ FAST Direct Method")
        print(f"Distance: {self.distance_km:.1f} km")
        print(f"Nodes: {self.n_nodes} (optimized for speed)")
        print(f"Estimated time: {self.tf_estimate/60:.1f} minutes")
        
    def create_simple_guess(self):
        """Very simple initial guess"""
        n_vars = self.n_nodes * 9 + 1
        x_guess = np.zeros(n_vars)
        
        dx = self.xf[0] - self.x0[0]
        dy = self.xf[1] - self.x0[1] 
        dh = self.xf[2] - self.x0[2]
        
        heading = np.arctan2(dy, dx)
        if heading < 0:
            heading += 2*np.pi
        
        for i in range(self.n_nodes):
            idx = i * 9
            alpha = i / (self.n_nodes - 1)
            
            # Simple linear interpolation
            x_guess[idx:idx+9] = [
                self.x0[0] + alpha * dx,                    # x
                self.x0[1] + alpha * dy,                    # y  
                self.x0[2] + alpha * dh,                    # h
                self.x0[3],                                 # v
                heading,                                    # psi
                self.x0[5] * (1 - 0.1 * alpha),           # m (10% fuel)
                np.sign(dh) * 0.02 if dh != 0 else 0.0,   # gamma
                0.0,                                        # mu
                0.7                                         # delta
            ]
        
        x_guess[-1] = self.tf_estimate
        return x_guess
    
    def create_simple_bounds(self):
        """Simple bounds"""
        bounds = []
        
        for i in range(self.n_nodes):
            bounds.extend([
                (0, 50),                     # x
                (35, 65),                    # y
                (5000, 12000),               # h
                (150, 300),                  # v
                (-2*np.pi, 2*np.pi),         # psi
                (self.x0[5]*0.7, self.x0[5]), # m
                (-0.1, 0.1),                 # gamma
                (-np.pi/6, np.pi/6),         # mu
                (0.3, 0.9)                   # delta
            ])
        
        bounds.append((self.tf_estimate*0.7, self.tf_estimate*1.5))
        return bounds
    
    def simple_objective(self, x):
        """Ultra-simple objective - just minimize fuel"""
        n = self.n_nodes
        
        initial_mass = x[5]
        final_mass = x[(n-1)*9 + 5]
        fuel_consumed = initial_mass - final_mass
        
        if fuel_consumed <= 0:
            return 1e6
        
        # Terminal penalty
        final_x = x[(n-1)*9]
        final_y = x[(n-1)*9 + 1]
        final_h = x[(n-1)*9 + 2]
        
        terminal_penalty = 100 * (
            (final_x - self.xf[0])**2 +
            (final_y - self.xf[1])**2 +
            (final_h - self.xf[2])**2 / 1e6
        )
        
        return fuel_consumed + terminal_penalty
    
    def simple_constraints(self, x):
        """Simple constraints - just dynamics"""
        n = self.n_nodes
        tf = x[-1]
        dt = tf / self.n_segments
        
        constraints = []
        
        # Initial conditions
        for i in range(6):
            constraints.append(x[i] - self.x0[i])
        
        # Simple Euler integration for speed
        for i in range(self.n_segments):
            idx1 = i * 9
            idx2 = (i+1) * 9
            
            state1 = x[idx1:idx1+6]
            controls1 = x[idx1+6:idx1+9]
            state2 = x[idx2:idx2+6]
            
            f1 = self.aircraft.dynamics(0, state1, controls1)
            
            for j in range(6):
                predicted = state1[j] + dt * f1[j]
                constraints.append(state2[j] - predicted)
        
        # Terminal conditions
        final_idx = (n-1) * 9
        constraints.extend([
            x[final_idx] - self.xf[0],
            x[final_idx+1] - self.xf[1], 
            x[final_idx+2] - self.xf[2]
        ])
        
        return np.array(constraints)
    
    def solve_fast(self):
        """Fast solve with minimal complexity"""
        print("\n⚡ Starting FAST optimization...")
        
        x_guess = self.create_simple_guess()
        bounds = self.create_simple_bounds()
        
        print(f"Problem size: {len(x_guess)} vars, {len(self.simple_constraints(x_guess))} constraints")
        
        try:
            result = minimize(
                self.simple_objective,
                x_guess,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': self.simple_constraints},
                options={
                    'maxiter': 100,  # Limited iterations for speed
                    'ftol': 1e-4,    # Looser tolerance
                    'eps': 1e-3,     # Larger step size
                    'disp': True
                }
            )
            
            if result.success or (hasattr(result, 'fun') and result.fun < 1e4):
                return self.extract_fast_solution(result.x)
            else:
                print(f"❌ Fast optimization failed: {result.message}")
                return None, None, None, False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None, False
    
    def extract_fast_solution(self, x_opt):
        """Extract solution"""
        n = self.n_nodes
        tf = x_opt[-1]
        
        t = np.linspace(0, tf, n)
        states = np.zeros((n, 6))
        controls = np.zeros((n, 3))
        
        for i in range(n):
            idx = i * 9
            states[i] = x_opt[idx:idx+6]
            controls[i] = x_opt[idx+6:idx+9]
        
        fuel_consumed = states[0,5] - states[-1,5]
        pos_error = np.sqrt((states[-1,0] - self.xf[0])**2 + (states[-1,1] - self.xf[1])**2)
        alt_error = abs(states[-1,2] - self.xf[2])
        
        print(f"✅ Fast solution found!")
        print(f"Flight time: {tf/60:.1f} minutes")
        print(f"Fuel consumed: {fuel_consumed:.1f} kg")
        print(f"Position error: {pos_error:.6f} degrees")
        print(f"Altitude error: {alt_error:.1f} meters")
        
        success = (pos_error < 0.02) and (alt_error < 300) and (fuel_consumed > 0)
        return t, states, controls, success


def solve_flight_plan(flight_plan_number):
    """Solve using fast method"""
    
    flight_plans = {
        1: {
            'name': 'Flight 1',
            'initial': [5, 40, 8000, 210, 0, 68000],
            'target': [32, 40, 8000]
        },
        2: {
            'name': 'Flight 2',
            'initial': [30, 55, 7000, 220, np.radians(40), 67000],
            'target': [15, 40, 9000]
        },
        3: {
            'name': 'Flight 3', 
            'initial': [32, 45, 8000, 210, np.radians(180), 65000],
            'target': [5, 45, 7000]
        }
    }
    
    plan = flight_plans[flight_plan_number]
    
    print(f"\n{'='*50}")
    print(f"⚡ SOLVING {plan['name'].upper()} - FAST METHOD")
    print(f"{'='*50}")
    
    aircraft = AircraftModel()
    optimizer = FastDirectMethod(aircraft, plan['initial'], plan['target'])
    
    result = optimizer.solve_fast()
    
    if result[0] is not None:
        t, states, controls, success = result
        
        if success:
            fuel_consumed = states[0,5] - states[-1,5]
            plot_fast_results(t, states, controls, flight_plan_number)
            
            return {
                'success': True,
                'time': t[-1],
                'fuel': fuel_consumed
            }
    
    return {'success': False, 'time': 0, 'fuel': 0}


def plot_fast_results(t, states, controls, flight_plan_number):
    """Quick plotting"""
    t_min = t / 60
    x, y, h, v, psi, m = states.T
    gamma, mu, delta = controls.T
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Flight Plan {flight_plan_number} - Fast Results', fontsize=14)
    
    # 1. Trajectory
    axes[0,0].plot(x, y, 'b-', linewidth=2)
    axes[0,0].plot(x[0], y[0], 'go', markersize=8)
    axes[0,0].plot(x[-1], y[-1], 'ro', markersize=8)
    axes[0,0].set_title('Trajectory')
    axes[0,0].grid(True)
    
    # 2. Altitude
    axes[0,1].plot(t_min, h/1000, 'b-', linewidth=2)
    axes[0,1].set_title('Altitude [km]')
    axes[0,1].grid(True)
    
    # 3. Mass
    axes[0,2].plot(t_min, m/1000, 'purple', linewidth=2)
    axes[0,2].set_title('Mass [tonnes]')
    axes[0,2].grid(True)
    
    # 4. Throttle
    axes[1,0].plot(t_min, delta, 'brown', linewidth=2)
    axes[1,0].set_title('Throttle')
    axes[1,0].set_ylim([0, 1])
    axes[1,0].grid(True)
    
    # 5. Flight path angle
    axes[1,1].plot(t_min, np.degrees(gamma), 'cyan', linewidth=2)
    axes[1,1].set_title('Flight Path Angle [°]')
    axes[1,1].grid(True)
    
    # 6. Bank angle
    axes[1,2].plot(t_min, np.degrees(mu), 'red', linewidth=2)
    axes[1,2].set_title('Bank Angle [°]')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'flight_plan_{flight_plan_number}_fast.png', dpi=200, bbox_inches='tight')
    plt.show()


def main():
    """Fast main execution"""
    print("⚡ FAST Aircraft Flight Optimization")
    print("🚀 Optimized for SPEED - 15 nodes only")
    print("=" * 50)
    
    results = {}
    
    for flight_num in [1, 2, 3]:
        result = solve_flight_plan(flight_num)
        results[flight_num] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("⚡ FAST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Flight':<8} | {'Status':<12} | {'Time (min)':<12} | {'Fuel (kg)':<12}")
    print("-" * 60)
    
    for flight_num, data in results.items():
        if data['success']:
            status = "✅ SUCCESS"
            time_str = f"{data['time']/60:.1f}"
            fuel_str = f"{data['fuel']:.1f}"
        else:
            status = "❌ FAILED"
            time_str = "N/A"
            fuel_str = "N/A"
        
        print(f"{flight_num:<8} | {status:<12} | {time_str:<12} | {fuel_str:<12}")


if __name__ == "__main__":
    main()