class ImprovedDirectCollocation:
    def __init__(self, aircraft_model, initial_state, target_state, n_nodes=25):
        self.aircraft = aircraft_model
        self.x0 = initial_state
        self.xf = target_state
        self.n_nodes = n_nodes
        
        # Calculate distance and estimate flight time
        dx = self.xf[0] - self.x0[0]
        dy = self.xf[1] - self.x0[1]
        self.dist_km = np.sqrt(dx**2 + dy**2) * 111.32
        self.tf_guess = (self.dist_km * 1000) / 220  # Conservative speed estimate
        
        print(f"Flight distance: {self.dist_km:.1f} km")
        print(f"Estimated flight time: {self.tf_guess/60:.1f} minutes")
    
    def create_smooth_initial_guess(self):
        """Create a smooth, physically realistic initial guess"""
        n = self.n_nodes
        x_guess = np.zeros(n * 9 + 1)
        
        # Calculate direct path parameters
        dx = self.xf[0] - self.x0[0]
        dy = self.xf[1] - self.x0[1]
        dh = self.xf[2] - self.x0[2]
        
        # Calculate direct heading (handle angle wrapping)
        direct_heading = np.arctan2(dy, dx)
        if direct_heading < 0:
            direct_heading += 2*np.pi
        
        # Create smooth interpolation for states
        for i in range(n):
            idx = i * 9
            alpha = i / (n - 1)
            
            # Position: simple linear interpolation
            x_guess[idx] = self.x0[0] + alpha * dx
            x_guess[idx + 1] = self.x0[1] + alpha * dy
            x_guess[idx + 2] = self.x0[2] + alpha * dh
            
            # Speed: constant
            x_guess[idx + 3] = self.x0[3]
            
            # Heading: constant towards target
            x_guess[idx + 4] = direct_heading
            
            # Mass: realistic linear decrease
            fuel_rate = 1.2  # kg/s (conservative)
            x_guess[idx + 5] = self.x0[5] - alpha * fuel_rate * self.tf_guess
            
            # Controls: minimal and smooth
            x_guess[idx + 6] = 0.01 * np.sign(dh) if dh != 0 else 0.0  # Gentle climb/descent
            x_guess[idx + 7] = 0.0  # No bank angle initially
            x_guess[idx + 8] = 0.65  # Moderate throttle setting
        
        # Time estimate
        x_guess[-1] = self.tf_guess
        
        return x_guess
    
    def smooth_trajectory_objective(self, x):
        """Objective function focused on smooth trajectories"""
        n = self.n_nodes
        tf = x[-1]
        dt = tf / (n - 1)
        
        # Main costs
        fuel_cost = 0.0
        time_cost = 0.01 * tf
        smoothness_cost = 0.0
        terminal_cost = 0.0
        
        # Fuel consumption (realistic calculation)
        for i in range(n-1):
            idx = i * 9
            m_i = x[idx + 5]
            m_next = x[idx + 9 + 5]
            fuel_used = m_i - m_next
            fuel_cost += max(fuel_used, 0.1 * dt)  # Ensure positive fuel flow
        
        # Control smoothness penalties
        for i in range(n):
            idx = i * 9
            gamma_i = x[idx + 6]
            mu_i = x[idx + 7]
            delta_i = x[idx + 8]
            
            # Penalize extreme controls
            smoothness_cost += 50.0 * (gamma_i**2 + mu_i**2)
            
            # Throttle efficiency penalty
            if delta_i > 0.8:
                smoothness_cost += 100.0 * (delta_i - 0.8)**2
            elif delta_i < 0.5:
                smoothness_cost += 100.0 * (0.5 - delta_i)**2
        
        # Control rate smoothness (prevent oscillations)
        control_rate_penalty = 0.0
        for i in range(n-1):
            idx_i = i * 9
            idx_next = (i + 1) * 9
            
            dgamma = (x[idx_next + 6] - x[idx_i + 6]) / dt
            dmu = (x[idx_next + 7] - x[idx_i + 7]) / dt
            ddelta = (x[idx_next + 8] - x[idx_i + 8]) / dt
            
            control_rate_penalty += 10.0 * dt * (dgamma**2 + dmu**2 + ddelta**2)
        
        # Trajectory smoothness (heading rate)
        heading_smoothness = 0.0
        for i in range(1, n-1):
            idx_prev = (i-1) * 9
            idx_curr = i * 9
            idx_next = (i+1) * 9
            
            psi_prev = x[idx_prev + 4]
            psi_curr = x[idx_curr + 4]
            psi_next = x[idx_next + 4]
            
            # Handle angle wrapping
            def angle_diff(a1, a2):
                diff = a1 - a2
                while diff > np.pi:
                    diff -= 2*np.pi
                while diff < -np.pi:
                    diff += 2*np.pi
                return diff
            
            # Second derivative of heading (curvature)
            d_psi_1 = angle_diff(psi_curr, psi_prev)
            d_psi_2 = angle_diff(psi_next, psi_curr)
            curvature = (d_psi_2 - d_psi_1) / dt**2
            
            heading_smoothness += 100.0 * curvature**2
        
        # Terminal position cost
        end_idx = (n-1) * 9
        terminal_cost = 5000.0 * (
            (x[end_idx] - self.xf[0])**2 + 
            (x[end_idx + 1] - self.xf[1])**2 + 
            (x[end_idx + 2] - self.xf[2])**2 / 1e6
        )
        
        total_cost = (fuel_cost + time_cost + smoothness_cost + 
                     control_rate_penalty + heading_smoothness + terminal_cost)
        
        return total_cost
    
    def enhanced_constraints(self, x):
        """Enhanced constraint function with better integration"""
        n = self.n_nodes
        tf = x[-1]
        dt = tf / (n - 1)
        
        constraints = []
        
        # Initial conditions
        for i in range(6):
            constraints.append(x[i] - self.x0[i])
        
        # Dynamics constraints using trapezoidal integration
        for i in range(n-1):
            idx_i = i * 9
            idx_ip1 = idx_i + 9
            
            state_i = x[idx_i:idx_i+6]
            controls_i = x[idx_i+6:idx_i+9]
            state_ip1 = x[idx_ip1:idx_ip1+6]
            controls_ip1 = x[idx_ip1+6:idx_ip1+9]
            
            # Dynamics at both points
            deriv_i = self.aircraft.dynamics(0, state_i, controls_i)
            deriv_ip1 = self.aircraft.dynamics(0, state_ip1, controls_ip1)
            
            # Trapezoidal integration
            for j in range(6):
                predicted = state_i[j] + dt * 0.5 * (deriv_i[j] + deriv_ip1[j])
                constraints.append(state_ip1[j] - predicted)
        
        # Terminal constraints
        end_idx = (n-1) * 9
        for i in range(3):  # Only x, y, h
            constraints.append(x[end_idx + i] - self.xf[i])
        
        # Mass decrease constraints (prevent mass increase)
        for i in range(n-1):
            idx_i = i * 9
            idx_ip1 = idx_i + 9
            mass_decrease = x[idx_i + 5] - x[idx_ip1 + 5]
            # Ensure reasonable fuel consumption rate
            constraints.append(mass_decrease - dt * 0.3)  # Min 0.3 kg/s
            constraints.append(dt * 3.0 - mass_decrease)    # Max 3.0 kg/s
        
        return np.array(constraints)
    
    def create_bounds(self):
        """Create realistic bounds for all variables"""
        n = self.n_nodes
        bounds = []
        
        # Calculate reasonable bounds for mass
        total_distance_km = self.dist_km
        estimated_fuel = min(total_distance_km * 0.6, self.x0[5] * 0.15)  # Conservative
        min_final_mass = max(self.x0[5] - estimated_fuel, self.x0[5] * 0.8)
        
        for i in range(n):
            progress = i / (n - 1)
            
            # Expected mass at this point
            expected_mass = self.x0[5] - progress * estimated_fuel
            mass_tolerance = self.x0[5] * 0.05
            
            # State bounds
            bounds.extend([
                # Position bounds (reasonable for European flights)
                (-10, 60),                     # Longitude
                (35, 70),                      # Latitude
                (5000, 13000),                 # Altitude
                (180, 280),                    # Speed
                (0, 2*np.pi),                  # Heading
                (max(min_final_mass, expected_mass - mass_tolerance), 
                 min(self.x0[5], expected_mass + mass_tolerance))  # Mass
            ])
            
            # Control bounds (conservative for smooth flight)
            bounds.extend([
                (-0.04, 0.04),        # Flight path angle: ±2.3°
                (-np.pi/20, np.pi/20), # Bank angle: ±9°
                (0.5, 0.8)            # Throttle setting
            ])
        
        # Time bounds
        bounds.append((self.tf_guess * 0.8, self.tf_guess * 1.3))
        
        return bounds
    
    def solve(self):
        """Solve the optimization problem with enhanced convergence"""
        print("Setting up optimization problem...")
        
        # Create smooth initial guess
        x_guess = self.create_smooth_initial_guess()
        bounds = self.create_bounds()
        
        print("Starting optimization...")
        
        # Use sequential approach for better convergence
        
        # Phase 1: Relaxed problem
        print("Phase 1: Solving relaxed problem...")
        result1 = minimize(
            self.smooth_trajectory_objective,
            x_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': self.enhanced_constraints},
            options={
                'maxiter': 300,
                'ftol': 1e-5,
                'eps': 1e-4,
                'disp': True,
                'finite_diff_rel_step': 1e-5
            }
        )
        
        if not result1.success:
            print("Phase 1 failed, trying alternative approach...")
            # Try with L-BFGS-B (bound-constrained only)
            result1 = minimize(
                lambda x: self.smooth_trajectory_objective(x) + self._penalty_for_constraints(x),
                x_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'ftol': 1e-6,
                    'gtol': 1e-6,
                    'disp': True
                }
            )
        
        # Phase 2: Refine solution
        if hasattr(result1, 'x'):
            print("Phase 2: Refining solution...")
            result2 = minimize(
                self.smooth_trajectory_objective,
                result1.x,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': self.enhanced_constraints},
                options={
                    'maxiter': 200,
                    'ftol': 1e-8,
                    'eps': 1e-6,
                    'disp': True
                }
            )
            
            if result2.success:
                result_final = result2
            else:
                result_final = result1
        else:
            print("Optimization failed completely!")
            return None, None, None, False
        
        # Extract solution
        return self._extract_solution(result_final)
    
    def _penalty_for_constraints(self, x):
        """Penalty function for constraint violations"""
        constraints = self.enhanced_constraints(x)
        penalty = 1000.0 * np.sum(constraints**2)
        return penalty
    
    def _extract_solution(self, result):
        """Extract and validate the solution"""
        if not hasattr(result, 'x'):
            return None, None, None, False
        
        x_opt = result.x
        tf = x_opt[-1]
        
        # Extract states and controls
        n = self.n_nodes
        t = np.linspace(0, tf, n)
        states = np.zeros((n, 6))
        controls = np.zeros((n, 3))
        
        for i in range(n):
            idx = i * 9
            states[i] = x_opt[idx:idx+6]
            controls[i] = x_opt[idx+6:idx+9]
        
        # Post-process controls to ensure smoothness
        controls = self._smooth_controls(controls)
        
        # Validation
        final_pos_error = np.sqrt((states[-1, 0] - self.xf[0])**2 + 
                                 (states[-1, 1] - self.xf[1])**2)
        alt_error = abs(states[-1, 2] - self.xf[2])
        
        success = (final_pos_error < 0.05 and alt_error < 200)
        
        print(f"Optimization completed:")
        print(f"  Position error: {final_pos_error:.4f} degrees ({final_pos_error*111:.1f} km)")
        print(f"  Altitude error: {alt_error:.1f} meters")
        print(f"  Flight time: {tf/60:.1f} minutes")
        print(f"  Fuel consumption: {states[0,5] - states[-1,5]:.1f} kg")
        print(f"  Success: {success}")
        
        return t, states, controls, success
    
    def _smooth_controls(self, controls):
        """Apply post-processing smoothing to controls"""
        n = len(controls)
        smoothed = np.copy(controls)
        
        # Simple moving average for throttle
        window = min(5, n//3)
        if window >= 3:
            for i in range(window//2, n - window//2):
                start = i - window//2
                end = i + window//2 + 1
                smoothed[i, 2] = np.mean(controls[start:end, 2])
        
        # Limit control rates
        max_gamma_rate = 0.02  # rad/s
        max_mu_rate = 0.05     # rad/s
        max_delta_rate = 0.1   # per second
        
        dt = 1.0  # Approximate dt
        
        for i in range(1, n):
            # Flight path angle
            dgamma = smoothed[i, 0] - smoothed[i-1, 0]
            if abs(dgamma/dt) > max_gamma_rate:
                smoothed[i, 0] = smoothed[i-1, 0] + np.sign(dgamma) * max_gamma_rate * dt
            
            # Bank angle
            dmu = smoothed[i, 1] - smoothed[i-1, 1]
            if abs(dmu/dt) > max_mu_rate:
                smoothed[i, 1] = smoothed[i-1, 1] + np.sign(dmu) * max_mu_rate * dt
            
            # Throttle
            ddelta = smoothed[i, 2] - smoothed[i-1, 2]
            if abs(ddelta/dt) > max_delta_rate:
                smoothed[i, 2] = smoothed[i-1, 2] + np.sign(ddelta) * max_delta_rate * dt
        
        return smoothed