import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import time
import json
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class FlightPlan:
    """Flight plan data structure"""
    name: str
    initial: np.ndarray
    target: np.ndarray

class AircraftModel:
    """Aircraft dynamics model for B737-800"""
    def __init__(self):
        # Environmental constants
        self.g = 9.81  # Gravity [m/s^2]
        self.rho_0 = 1.225  # Sea level air density [kg/m^3]
        
        # Aircraft parameters
        self.S = 124.65  # Wing area [m^2]
        
        # Aerodynamic coefficients
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
        
        # Wind field coefficients
        self.wx_coef = np.array([-21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001])
        self.wy_coef = np.array([-65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227])

    def air_density(self, h):
        """Air density as function of altitude [kg/m^3]"""
        return self.rho_0 * (1 - 2.2257e-5 * h) ** 4.2561

    def thrust_max(self, h):
        """Maximum thrust as function of altitude [N]"""
        return self.CT1 * self.CT2_1 * (1 - (3.28 * h) / self.CT2_2) + self.CT2_3 * (3.28 * h)**2

    def eta(self, v):
        """Fuel consumption rate parameter"""
        return (self.Cf3 / 60000) * (1 + (1.943 * v) / self.Cf3)

    def fuel_flow(self, delta, thr_max, eta):
        """Fuel flow rate [kg/s]"""
        flow = delta * thr_max * eta * self.Cf1 / 1e6
        return max(0.001, flow)

    def wind_speed(self, lon, lat):
        """Wind speed components at given position [m/s]"""
        terms = [1, lon, lat, lon*lat, lon**2, lat**2, 
                lon**2*lat, lon*lat**2, lon**2*lat**2]
        
        W_x = sum(c*t for c, t in zip(self.wx_coef, terms))
        W_y = sum(c*t for c, t in zip(self.wy_coef, terms))
        
        return W_x, W_y

    def lift_coefficient(self, m, rho, v, mu):
        """Lift coefficient"""
        if v < 50:  # Prevent division by zero
            return 0
        return (2 * m * self.g) / (rho * self.S * v**2 * np.cos(mu))

    def drag_coefficient(self, CL):
        """Drag coefficient"""
        return self.C_D0 + self.k * CL**2

    def dynamics(self, state, control):
        """Compute state derivatives"""
        x, y, h, v, psi, m = state
        gamma, mu, delta = control
        
        # Clamp controls
        gamma = np.clip(gamma, -0.1, 0.1)
        mu = np.clip(mu, -np.pi/6, np.pi/6)
        delta = np.clip(delta, 0.3, 0.9)
        
        # Environmental conditions
        rho = self.air_density(h)
        W_x, W_y = self.wind_speed(x, y)
        
        # Aerodynamics
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
        
        return np.array([x_dot, y_dot, h_dot, v_dot, psi_dot, m_dot])


class ImprovedDPSolver:
    """Improved Dynamic Programming solver with better numerical stability"""
    
    def __init__(self, aircraft_model: AircraftModel, flight_plan: FlightPlan):
        self.aircraft = aircraft_model
        self.plan = flight_plan
        self.x0 = flight_plan.initial
        self.xf = flight_plan.target
        
        # Calculate flight parameters
        dx = self.xf[0] - self.x0[0]
        dy = self.xf[1] - self.x0[1]
        self.distance_km = np.sqrt(dx**2 + dy**2) * 111
        self.initial_heading = np.arctan2(dy, dx)
        
        # Time discretization
        self.n_time_steps = 30  # Number of time steps
        self.tf_estimate = (self.distance_km / 700) * 3600  # Estimate flight time
        self.dt = self.tf_estimate / (self.n_time_steps - 1)
        
        # State space discretization (adaptive based on problem)
        self._create_adaptive_grids()
        
        # Control discretization
        self.n_controls = 3  # Reduced for computational efficiency
        self.gamma_values = np.array([-0.05, 0.0, 0.05])  # Climb/descent angles
        self.mu_values = np.array([-0.1, 0.0, 0.1])  # Bank angles
        self.delta_values = np.array([0.5, 0.7, 0.9])  # Throttle settings
        
        print(f"\n🎯 Improved DP Solver Initialized")
        print(f"Flight: {flight_plan.name}")
        print(f"Distance: {self.distance_km:.1f} km")
        print(f"Estimated time: {self.tf_estimate/60:.1f} minutes")
        print(f"Time steps: {self.n_time_steps}")
        print(f"State grid size: {self._calculate_grid_size():,} states")
        
    def _create_adaptive_grids(self):
        """Create adaptive grids focused around the flight path"""
        # Position grids - focused around start and end points
        x_range = abs(self.xf[0] - self.x0[0])
        y_range = abs(self.xf[1] - self.x0[1])
        
        self.x_grid = np.linspace(
            min(self.x0[0], self.xf[0]) - 0.2 * x_range,
            max(self.x0[0], self.xf[0]) + 0.2 * x_range,
            8
        )
        
        self.y_grid = np.linspace(
            min(self.x0[1], self.xf[1]) - 0.2 * y_range,
            max(self.x0[1], self.xf[1]) + 0.2 * y_range,
            8
        )
        
        # Altitude grid - focused around cruise altitude
        h_min = min(self.x0[2], self.xf[2]) - 1000
        h_max = max(self.x0[2], self.xf[2]) + 1000
        self.h_grid = np.linspace(h_min, h_max, 5)
        
        # Velocity grid - typical cruise speeds
        self.v_grid = np.linspace(180, 250, 5)
        
        # Heading grid - focused around direct path
        psi_range = np.pi / 3  # ±60 degrees from direct path
        self.psi_grid = np.linspace(
            self.initial_heading - psi_range,
            self.initial_heading + psi_range,
            6
        )
        
        # Mass grid - expected fuel consumption
        max_fuel_burn = 0.15 * self.x0[5]  # Max 15% fuel burn
        self.m_grid = np.linspace(
            self.x0[5] - max_fuel_burn,
            self.x0[5],
            4
        )
        
    def _calculate_grid_size(self):
        """Calculate total number of grid points"""
        return (len(self.x_grid) * len(self.y_grid) * len(self.h_grid) * 
                len(self.v_grid) * len(self.psi_grid) * len(self.m_grid))
    
    def _state_to_indices(self, state):
        """Convert continuous state to grid indices"""
        x, y, h, v, psi, m = state
        
        # Find nearest grid points
        ix = np.searchsorted(self.x_grid, x)
        iy = np.searchsorted(self.y_grid, y)
        ih = np.searchsorted(self.h_grid, h)
        iv = np.searchsorted(self.v_grid, v)
        ipsi = np.searchsorted(self.psi_grid, psi)
        im = np.searchsorted(self.m_grid, m)
        
        # Ensure within bounds
        ix = np.clip(ix, 0, len(self.x_grid) - 1)
        iy = np.clip(iy, 0, len(self.y_grid) - 1)
        ih = np.clip(ih, 0, len(self.h_grid) - 1)
        iv = np.clip(iv, 0, len(self.v_grid) - 1)
        ipsi = np.clip(ipsi, 0, len(self.psi_grid) - 1)
        im = np.clip(im, 0, len(self.m_grid) - 1)
        
        return (ix, iy, ih, iv, ipsi, im)
    
    def _indices_to_state(self, indices):
        """Convert grid indices to state"""
        ix, iy, ih, iv, ipsi, im = indices
        return np.array([
            self.x_grid[ix],
            self.y_grid[iy],
            self.h_grid[ih],
            self.v_grid[iv],
            self.psi_grid[ipsi],
            self.m_grid[im]
        ])
    
    def _propagate_state(self, state, control, dt):
        """Propagate state forward using RK4 integration"""
        k1 = self.aircraft.dynamics(state, control)
        k2 = self.aircraft.dynamics(state + 0.5 * dt * k1, control)
        k3 = self.aircraft.dynamics(state + 0.5 * dt * k2, control)
        k4 = self.aircraft.dynamics(state + dt * k3, control)
        
        new_state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Ensure valid state
        new_state[3] = np.clip(new_state[3], 100, 300)  # Velocity
        new_state[4] = np.arctan2(np.sin(new_state[4]), np.cos(new_state[4]))  # Wrap heading
        new_state[5] = max(new_state[5], self.x0[5] * 0.7)  # Min mass
        
        return new_state
    
    def _stage_cost(self, state, control, next_state, dt):
        """Stage cost function"""
        # Fuel consumption
        fuel_rate = (state[5] - next_state[5]) / dt
        
        # Time penalty
        time_cost = 0.01  # Small time penalty
        
        # Control effort penalty (smooth controls)
        control_cost = 0.001 * (control[0]**2 + control[1]**2 + (control[2] - 0.7)**2)
        
        return fuel_rate + time_cost + control_cost
    
    def _terminal_cost(self, state):
        """Terminal cost function"""
        x, y, h = state[:3]
        
        # Position error in km
        pos_error = np.sqrt((x - self.xf[0])**2 + (y - self.xf[1])**2) * 111
        
        # Altitude error in km
        alt_error = abs(h - self.xf[2]) / 1000
        
        # Quadratic penalty
        return 1000 * (pos_error**2 + alt_error**2)
    
    def solve_dp(self):
        """Solve using Dynamic Programming with value iteration"""
        print("\n🚀 Starting DP solution...")
        start_time = time.time()
        
        # Initialize value function
        shape = (len(self.x_grid), len(self.y_grid), len(self.h_grid),
                 len(self.v_grid), len(self.psi_grid), len(self.m_grid),
                 self.n_time_steps)
        
        V = np.full(shape, np.inf)
        policy = {}
        
        # Terminal condition
        print("Setting terminal conditions...")
        for ix in range(len(self.x_grid)):
            for iy in range(len(self.y_grid)):
                for ih in range(len(self.h_grid)):
                    for iv in range(len(self.v_grid)):
                        for ipsi in range(len(self.psi_grid)):
                            for im in range(len(self.m_grid)):
                                state = self._indices_to_state((ix, iy, ih, iv, ipsi, im))
                                V[ix, iy, ih, iv, ipsi, im, -1] = self._terminal_cost(state)
        
        # Backward induction
        print("Performing backward induction...")
        for t in range(self.n_time_steps - 2, -1, -1):
            print(f"Time step {self.n_time_steps - t}/{self.n_time_steps}", end='\r')
            
            for ix in range(len(self.x_grid)):
                for iy in range(len(self.y_grid)):
                    for ih in range(len(self.h_grid)):
                        for iv in range(len(self.v_grid)):
                            for ipsi in range(len(self.psi_grid)):
                                for im in range(len(self.m_grid)):
                                    state = self._indices_to_state((ix, iy, ih, iv, ipsi, im))
                                    
                                    best_value = np.inf
                                    best_control = None
                                    
                                    # Try all control combinations
                                    for gamma in self.gamma_values:
                                        for mu in self.mu_values:
                                            for delta in self.delta_values:
                                                control = np.array([gamma, mu, delta])
                                                
                                                # Propagate state
                                                next_state = self._propagate_state(state, control, self.dt)
                                                
                                                # Get next state indices
                                                next_indices = self._state_to_indices(next_state)
                                                
                                                # Stage cost
                                                stage_cost = self._stage_cost(state, control, next_state, self.dt)
                                                
                                                # Future cost (with interpolation)
                                                try:
                                                    future_cost = V[next_indices[0], next_indices[1], next_indices[2],
                                                                   next_indices[3], next_indices[4], next_indices[5], t+1]
                                                except:
                                                    future_cost = np.inf
                                                
                                                total_cost = stage_cost * self.dt + future_cost
                                                
                                                if total_cost < best_value:
                                                    best_value = total_cost
                                                    best_control = control
                                    
                                    V[ix, iy, ih, iv, ipsi, im, t] = best_value
                                    if best_control is not None:
                                        policy[(ix, iy, ih, iv, ipsi, im, t)] = best_control
        
        print(f"\n✅ DP solution completed in {time.time() - start_time:.1f} seconds")
        
        # Extract trajectory
        return self._extract_trajectory(V, policy)
    
    def _extract_trajectory(self, V, policy):
        """Extract optimal trajectory from policy"""
        print("\nExtracting optimal trajectory...")
        
        states = [self.x0.copy()]
        controls = []
        times = [0]
        
        current_state = self.x0.copy()
        
        for t in range(self.n_time_steps - 1):
            # Get current state indices
            indices = self._state_to_indices(current_state)
            key = indices + (t,)
            
            # Get optimal control from policy
            if key in policy:
                control = policy[key]
            else:
                # Default control if not in policy
                # Point towards target
                dx = self.xf[0] - current_state[0]
                dy = self.xf[1] - current_state[1]
                desired_heading = np.arctan2(dy, dx)
                heading_error = desired_heading - current_state[4]
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                
                # Simple proportional control
                mu = np.clip(0.5 * heading_error, -0.1, 0.1)
                
                # Altitude control
                dh = self.xf[2] - current_state[2]
                gamma = np.clip(0.001 * dh, -0.05, 0.05)
                
                control = np.array([gamma, mu, 0.7])
            
            controls.append(control)
            
            # Propagate state
            next_state = self._propagate_state(current_state, control, self.dt)
            states.append(next_state)
            times.append(times[-1] + self.dt)
            
            current_state = next_state
            
            # Check if close to target
            pos_error = np.sqrt((current_state[0] - self.xf[0])**2 + 
                               (current_state[1] - self.xf[1])**2) * 111
            if pos_error < 5:  # Within 5 km
                print(f"Target reached at t={times[-1]/60:.1f} minutes")
                break
        
        # Convert to arrays
        states = np.array(states)
        controls = np.array(controls)
        times = np.array(times)
        
        # Calculate metrics
        fuel_consumed = states[0, 5] - states[-1, 5]
        final_pos_error = np.sqrt((states[-1, 0] - self.xf[0])**2 + 
                                 (states[-1, 1] - self.xf[1])**2) * 111
        final_alt_error = abs(states[-1, 2] - self.xf[2])
        
        print(f"\n📊 Final Results:")
        print(f"Flight time: {times[-1]/60:.1f} minutes")
        print(f"Fuel consumed: {fuel_consumed:.1f} kg")
        print(f"Final position error: {final_pos_error:.1f} km")
        print(f"Final altitude error: {final_alt_error:.1f} m")
        
        return times, states, controls


def plot_improved_dp_results(t, states, controls, flight_num):
    """Plot improved DP results"""
    t_min = t / 60
    x, y, h, v, psi, m = states.T
    
    if len(controls) < len(t):
        controls = np.vstack([controls, controls[-1]])
    
    gamma, mu, delta = controls.T
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Flight Plan {flight_num} - Improved DP Results', fontsize=16)
    
    # Plot all subplots similar to before...
    # (keeping the same plotting code as before)
    
    # 1. 2D Trajectory
    ax = axes[0, 0]
    ax.plot(x, y, 'b-', linewidth=2, label='DP Trajectory')
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('Latitude [°]')
    ax.set_title('2D Trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Altitude Profile
    ax = axes[0, 1]
    ax.plot(t_min, h/1000, 'b-', linewidth=2)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title('Altitude Profile')
    ax.grid(True, alpha=0.3)
    
    # 3. Velocity Profile
    ax = axes[0, 2]
    ax.plot(t_min, v, 'g-', linewidth=2)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Velocity Profile')
    ax.grid(True, alpha=0.3)
    
    # 4. Heading Angle
    ax = axes[1, 0]
    ax.plot(t_min, np.degrees(psi), 'orange', linewidth=2)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Heading [°]')
    ax.set_title('Heading Angle')
    ax.grid(True, alpha=0.3)
    
    # 5. Mass Profile
    ax = axes[1, 1]
    ax.plot(t_min, m/1000, 'purple', linewidth=2)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Mass [tonnes]')
    ax.set_title('Aircraft Mass')
    ax.grid(True, alpha=0.3)
    
    # 6. Fuel Consumption Rate
    ax = axes[1, 2]
    if len(m) > 1:
        fuel_rate = -np.gradient(m, t)
        ax.plot(t_min[:-1], fuel_rate[:-1], 'brown', linewidth=2)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Fuel Rate [kg/s]')
    ax.set_title('Fuel Consumption Rate')
    ax.grid(True, alpha=0.3)
    
    # 7. Flight Path Angle
    ax = axes[2, 0]
    ax.plot(t_min, np.degrees(gamma), 'cyan', linewidth=2)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Flight Path Angle [°]')
    ax.set_title('Flight Path Angle (γ)')
    ax.grid(True, alpha=0.3)
    
    # 8. Bank Angle
    ax = axes[2, 1]
    ax.plot(t_min, np.degrees(mu), 'red', linewidth=2)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Bank Angle [°]')
    ax.set_title('Bank Angle (μ)')
    ax.grid(True, alpha=0.3)
    
    # 9. Throttle
    ax = axes[2, 2]
    ax.plot(t_min, delta, 'darkgreen', linewidth=2)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Throttle [-]')
    ax.set_title('Throttle Setting (δ)')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'flight_plan_{flight_num}_improved_dp.png', dpi=300, bbox_inches='tight')
    plt.show()


def solve_flight_improved_dp(flight_num):
    """Solve flight plan using improved DP"""
    
    flight_plans = {
        1: FlightPlan(
            name='Flight 1',
            initial=np.array([5, 40, 8000, 210, 0, 68000]),
            target=np.array([32, 40, 8000])
        ),
        2: FlightPlan(
            name='Flight 2',
            initial=np.array([30, 55, 7000, 220, np.radians(40), 67000]),
            target=np.array([15, 40, 9000])
        ),
        3: FlightPlan(
            name='Flight 3',
            initial=np.array([32, 45, 8000, 210, np.radians(180), 65000]),
            target=np.array([5, 45, 7000])
        )
    }
    
    plan = flight_plans[flight_num]
    
    print(f"\n{'='*60}")
    print(f"SOLVING {plan.name.upper()} - IMPROVED DYNAMIC PROGRAMMING")
    print(f"{'='*60}")
    
    # Create models
    aircraft = AircraftModel()
    solver = ImprovedDPSolver(aircraft, plan)
    
    # Solve
    times, states, controls = solver.solve_dp()
    
    # Plot
    plot_improved_dp_results(times, states, controls, flight_num)
    
    return {
        'flight': flight_num,
        'time': times[-1] / 60,
        'fuel': states[0, 5] - states[-1, 5],
        'final_pos_error': np.sqrt((states[-1, 0] - plan.target[0])**2 + 
                                  (states[-1, 1] - plan.target[1])**2) * 111,
        'final_alt_error': abs(states[-1, 2] - plan.target[2])
    }


def main():
    """Main function"""
    print("✈️  AIRCRAFT FLIGHT OPTIMIZATION - IMPROVED DYNAMIC PROGRAMMING")
    print("="*60)
    
    results = []
    
    for flight_num in [1, 2, 3]:
        try:
            result = solve_flight_improved_dp(flight_num)
            results.append(result)
        except Exception as e:
            print(f"❌ Error in Flight {flight_num}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*80}")
    print("IMPROVED DP RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Flight':<10} | {'Time (min)':<12} | {'Fuel (kg)':<12} | {'Pos Error (km)':<15} | {'Alt Error (m)':<15}")
    print("-"*80)
    
    for r in results:
        print(f"{r['flight']:<10} | {r['time']:<12.1f} | {r['fuel']:<12.1f} | "
              f"{r['final_pos_error']:<15.2f} | {r['final_alt_error']:<15.1f}")
    
    # Save results
    with open('improved_dp_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Results saved to 'improved_dp_results.json'")


if __name__ == "__main__":
    main()