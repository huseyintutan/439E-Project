import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import json
from aircraft_dynamics import aircraft_dynamics, compute_rho, compute_thrust_max, compute_eta, compute_lift_coeff, compute_drag_coeff, compute_fuel_flow, wind_field, wind_coeffs_x, wind_coeffs_y, g, S

class PMPSolver:
    def __init__(self, flight_plan):
        self.name = flight_plan["name"]
        self.initial = flight_plan["initial"]
        self.target = flight_plan["target"]
        
        self.x0 = self.initial["lambda"]
        self.y0 = self.initial["phi"]
        self.h0 = self.initial["h"]
        self.v0 = self.initial["v"]
        self.psi0 = np.radians(self.initial["psi"])
        self.m0 = self.initial["m"]
        
        self.xf = self.target["lambda"]
        self.yf = self.target["phi"]
        self.hf = self.target["h"]
        
        self.state0 = [self.x0, self.y0, self.h0, self.v0, self.psi0, self.m0]
        
        self.gamma_min, self.gamma_max = np.radians(-5), np.radians(5)
        self.mu_min, self.mu_max = np.radians(-25), np.radians(25)
        self.delta_min, self.delta_max = 0.1, 1.0
        
        self.weight_position = 1.0
        self.weight_height = 0.5
        self.weight_terminal_constraint = 1000.0
        
        self.n_gamma = 7
        self.n_mu = 7
        self.n_delta = 7
        
    def compute_wind_derivatives(self, x, y):
        dWx_dx = wind_coeffs_x[1] + wind_coeffs_x[3] * y + 2 * wind_coeffs_x[4] * x + \
                 2 * wind_coeffs_x[6] * x * y + wind_coeffs_x[7] * y**2 + 2 * wind_coeffs_x[8] * x * y**2
        
        dWx_dy = wind_coeffs_x[2] + wind_coeffs_x[3] * x + 2 * wind_coeffs_x[5] * y + \
                 wind_coeffs_x[6] * x**2 + 2 * wind_coeffs_x[7] * x * y + 2 * wind_coeffs_x[8] * x**2 * y
        
        dWy_dx = wind_coeffs_y[1] + wind_coeffs_y[3] * y + 2 * wind_coeffs_y[4] * x + \
                 2 * wind_coeffs_y[6] * x * y + wind_coeffs_y[7] * y**2 + 2 * wind_coeffs_y[8] * x * y**2
        
        dWy_dy = wind_coeffs_y[2] + wind_coeffs_y[3] * x + 2 * wind_coeffs_y[5] * y + \
                 wind_coeffs_y[6] * x**2 + 2 * wind_coeffs_y[7] * x * y + 2 * wind_coeffs_y[8] * x**2 * y
        
        return dWx_dx, dWx_dy, dWy_dx, dWy_dy
    
    def compute_density_derivatives(self, h):
        term = 1 - 2.2257e-5 * h
        if term <= 0:
            return 0.0
        
        drho_dh = 1.225 * 4.2561 * term**(4.2561-1) * (-2.2257e-5)
        return drho_dh
    
    def compute_thrust_derivatives(self, h):
        C0, C1, C2, C3 = [0.95, 146590, 53872, 3.0453e-11]
        dT_dh = -C0 * C1 * 3.28 / C2 + 2 * C3 * (3.28)**2 * h
        return dT_dh
    
    def compute_eta_derivatives(self, v):
        C_FC = [0.92958, 0.70057, 1068.1]
        deta_dv = C_FC[2] / 60000 * 1.943 / C_FC[2]
        return deta_dv
    
    def compute_CL_derivatives(self, m, rho, v, mu):
        dCL_drho = -2 * m * g / (rho**2 * S * v**2 * np.cos(mu))
        dCL_dv = -4 * m * g / (rho * S * v**3 * np.cos(mu))
        dCL_dmu = 2 * m * g * np.sin(mu) / (rho * S * v**2 * np.cos(mu)**2)
        dCL_dm = 2 * g / (rho * S * v**2 * np.cos(mu))
        
        return dCL_drho, dCL_dv, dCL_dmu, dCL_dm
    
    def compute_CD_derivatives(self, CL, dCL_drho, dCL_dv, dCL_dmu, dCL_dm):
        k = 0.035815
        dCD_dCL = 2 * k * CL
        dCD_drho = dCD_dCL * dCL_drho
        dCD_dv = dCD_dCL * dCL_dv
        dCD_dmu = dCD_dCL * dCL_dmu
        dCD_dm = dCD_dCL * dCL_dm
        
        return dCD_drho, dCD_dv, dCD_dmu, dCD_dm
    
    def compute_fuel_flow_derivatives(self, delta, h, v, dT_dh):
        C_FC0 = 0.92958
        T_max = compute_thrust_max(h)
        eta = compute_eta(v)
        deta_dv = self.compute_eta_derivatives(v)
        
        df_ddelta = T_max * eta * C_FC0
        df_dh = delta * dT_dh * eta * C_FC0
        df_dv = delta * T_max * deta_dv * C_FC0
        
        return df_ddelta, df_dh, df_dv
    
    def optimal_control(self, state, costate, t=None):
        x, y, h, v, psi, m = state
        px, py, ph, pv, ppsi, pm = costate
        
        gamma_range = np.linspace(self.gamma_min, self.gamma_max, self.n_gamma)
        mu_range = np.linspace(self.mu_min, self.mu_max, self.n_mu)
        delta_range = np.linspace(self.delta_min, self.delta_max, self.n_delta)
        
        min_H = float('inf')
        opt_control = [0, 0, 0]
        
        rho = compute_rho(h)
        T_max = compute_thrust_max(h)
        eta = compute_eta(v)
        
        for gamma in gamma_range:
            for mu in mu_range:
                for delta in delta_range:
                    control = [gamma, mu, delta]
                    dx, dy, dh, dv, dpsi, dm = aircraft_dynamics(None, state, control)
                    
                    f = compute_fuel_flow(delta, T_max, eta)
                    cost = 1 + (0.05 + f)
                    
                    H = cost + px * dx + py * dy + ph * dh + pv * dv + ppsi * dpsi + pm * dm
                    
                    if H < min_H:
                        min_H = H
                        opt_control = control
        
        return opt_control
    
    def costate_dynamics(self, state, costate, control, t=None):
        x, y, h, v, psi, m = state
        px, py, ph, pv, ppsi, pm = costate
        gamma, mu, delta = control
        
        rho = compute_rho(h)
        drho_dh = self.compute_density_derivatives(h)
        
        T_max = compute_thrust_max(h)
        dT_dh = self.compute_thrust_derivatives(h)
        
        eta = compute_eta(v)
        deta_dv = self.compute_eta_derivatives(v)
        
        dWx_dx, dWx_dy, dWy_dx, dWy_dy = self.compute_wind_derivatives(x, y)
        
        CL = compute_lift_coeff(m, rho, v, mu)
        dCL_drho, dCL_dv, dCL_dmu, dCL_dm = self.compute_CL_derivatives(m, rho, v, mu)
        
        CD = compute_drag_coeff(CL)
        dCD_drho, dCD_dv, dCD_dmu, dCD_dm = self.compute_CD_derivatives(CL, dCL_drho, dCL_dv, dCL_dmu, dCL_dm)
        
        df_ddelta, df_dh, df_dv = self.compute_fuel_flow_derivatives(delta, h, v, dT_dh)
        
        # Eş-durum dinamikleri (PMP'nin -∂H/∂x kuralı)
        dpx = -(-px * dWx_dx - py * dWy_dx)  
        dpy = -(-px * dWx_dy - py * dWy_dy)  
        
        dph = -(-ph * 0 - pv * (delta * dT_dh / m - drho_dh * CD * S * v**2 / (2 * m)) - pm * df_dh)
        
        dpv = -(-px * np.cos(psi) * np.cos(gamma) -
               py * np.sin(psi) * np.cos(gamma) -
               ph * np.sin(gamma) -
               pv * (-rho * CD * S * v / m) -
               ppsi * (CL * S * rho / (2 * m * np.cos(gamma))) -
               pm * df_dv)
        
        dppsi = -(-px * (-v * np.sin(psi) * np.cos(gamma)) -
                 py * (v * np.cos(psi) * np.cos(gamma)))
        
        dpm = -(-pv * (delta * T_max / m**2 + CD * S * rho * v**2 / (2 * m**2)) -
               ppsi * (CL * S * rho * v / (2 * m**2 * np.cos(gamma))))
        
        return [dpx, dpy, dph, dpv, dppsi, dpm]
    
    def full_dynamics(self, t, full_state, tf=None):
        state = full_state[:6]
        costate = full_state[6:]
        
        control = self.optimal_control(state, costate)
        
        state_dot = aircraft_dynamics(t, state, control)
        costate_dot = self.costate_dynamics(state, costate, control)
        
        return np.concatenate([state_dot, costate_dot])
    
    def shooting_method(self, p0_guess, tf_guess):
        def objective(params):
            p0 = params[:-1]
            tf = params[-1]
            
            full_state0 = np.concatenate([self.state0, p0])
            
            sol = solve_ivp(
                self.full_dynamics, 
                [0, tf], 
                full_state0, 
                method='RK45', 
                rtol=1e-6, 
                atol=1e-6
            )
            
            if not sol.success:
                return 1e10
            
            final_state = sol.y[:6, -1]
            x_final, y_final, h_final = final_state[0], final_state[1], final_state[2]
            
            error = (x_final - self.xf)**2 + (y_final - self.yf)**2 + (h_final - self.hf)**2
            return error
        
        bounds = [(None, None)]*6 + [(100, 20000)]
        result = minimize(
            objective, 
            np.append(p0_guess, tf_guess), 
            method='SLSQP', 
            bounds=bounds, 
            options={'disp': True, 'maxiter': 100}
        )
        
        return result.x[:-1], result.x[-1], result.success
    
    def solve(self):
        p0_guess = np.zeros(6)
        tf_guess = 10000.0
        
        p0_opt, tf_opt, success = self.shooting_method(p0_guess, tf_guess)
        
        if not success:
            print(f"Uçuş {self.name} için optimizasyon başarısız!")
            return False
        
        full_state0 = np.concatenate([self.state0, p0_opt])
        
        t_eval = np.linspace(0, tf_opt, 1000)
        sol = solve_ivp(
            self.full_dynamics, 
            [0, tf_opt], 
            full_state0, 
            method='RK45', 
            t_eval=t_eval, 
            rtol=1e-6, 
            atol=1e-6
        )
        
        self.times = sol.t
        self.state_history = sol.y[:6].T
        self.costate_history = sol.y[6:].T
        
        self.controls = np.zeros((len(t_eval), 3))
        for i in range(len(t_eval)):
            self.controls[i] = self.optimal_control(self.state_history[i], self.costate_history[i])
        
        fuel_consumed = self.state0[5] - self.state_history[-1, 5]
        flight_time = tf_opt
        
        print(f"Uçuş {self.name} tamamlandı:")
        print(f"  Toplam yakıt tüketimi: {fuel_consumed:.2f} kg")
        print(f"  Toplam uçuş süresi: {flight_time:.2f} saniye")
        
        return True
    
    def plot_results(self):
        if not hasattr(self, 'state_history'):
            print("Önce 'solve' metodu çalıştırılmalıdır!")
            return
        
        plt.figure(figsize=(15, 10))
        
        # x-y düzleminde yörünge
        plt.subplot(3, 3, 1)
        plt.plot(self.state_history[:, 0], self.state_history[:, 1])
        plt.scatter(self.x0, self.y0, color='green', label='Başlangıç')
        plt.scatter(self.xf, self.yf, color='red', label='Hedef')
        plt.xlabel('Boylam (λ)')
        plt.ylabel('Enlem (φ)')
        plt.title('Yörünge (x-y düzlemi)')
        plt.legend()
        plt.grid(True)
        
        # Yükseklik - Zaman
        plt.subplot(3, 3, 2)
        plt.plot(self.times, self.state_history[:, 2])
        plt.xlabel('Zaman (s)')
        plt.ylabel('Yükseklik (m)')
        plt.title('Yükseklik - Zaman')
        plt.grid(True)
        
        # Hız - Zaman
        plt.subplot(3, 3, 3)
        plt.plot(self.times, self.state_history[:, 3])
        plt.xlabel('Zaman (s)')
        plt.ylabel('Hız (m/s)')
        plt.title('Hız - Zaman')
        plt.grid(True)
        
        # Kütle - Zaman
        plt.subplot(3, 3, 4)
        plt.plot(self.times, self.state_history[:, 5])
        plt.xlabel('Zaman (s)')
        plt.ylabel('Kütle (kg)')
        plt.title('Kütle - Zaman')
        plt.grid(True)
        
        # İtki - Zaman
        plt.subplot(3, 3, 5)
        thrust = np.zeros_like(self.times)
        for i in range(len(self.times)):
            h = self.state_history[i, 2]
            delta = self.controls[i, 2]
            thrust[i] = delta * compute_thrust_max(h)
        
        plt.plot(self.times, thrust)
        plt.xlabel('Zaman (s)')
        plt.ylabel('İtki (N)')
        plt.title('İtki - Zaman')
        plt.grid(True)
        
        # Throttle - Zaman
        plt.subplot(3, 3, 6)
        plt.plot(self.times, self.controls[:, 2])
        plt.xlabel('Zaman (s)')
        plt.ylabel('Throttle')
        plt.title('Throttle - Zaman')
        plt.grid(True)
        
        # Bank açısı - Zaman
        plt.subplot(3, 3, 7)
        plt.plot(self.times, np.degrees(self.controls[:, 1]))
        plt.xlabel('Zaman (s)')
        plt.ylabel('Bank Açısı (°)')
        plt.title('Bank Açısı - Zaman')
        plt.grid(True)
        
        # Uçuş yolu açısı - Zaman
        plt.subplot(3, 3, 8)
        plt.plot(self.times, np.degrees(self.controls[:, 0]))
        plt.xlabel('Zaman (s)')
        plt.ylabel('Uçuş Yolu Açısı (°)')
        plt.title('Uçuş Yolu Açısı - Zaman')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.name}_results.png", dpi=300)
        plt.show()

def main():
    with open('flight_plans.json', 'r') as f:
        flight_plans = json.load(f)
    
    results = []
    
    for flight_plan in flight_plans:
        print(f"\nUçuş {flight_plan['name']} çözülüyor...")
        
        solver = PMPSolver(flight_plan)
        success = solver.solve()
        
        if success:
            solver.plot_results()
            
            final_state = solver.state_history[-1]
            fuel_consumption = solver.state0[5] - final_state[5]
            flight_time = solver.times[-1]
            
            results.append({
                'name': flight_plan['name'],
                'fuel_consumption': fuel_consumption,
                'flight_time': flight_time,
                'final_position': (final_state[0], final_state[1], final_state[2])
            })
    
    print("\nÖzet Sonuçlar:")
    for result in results:
        print(f"Uçuş {result['name']}:")
        print(f"  Yakıt Tüketimi: {result['fuel_consumption']:.2f} kg")
        print(f"  Uçuş Süresi: {result['flight_time']:.2f} saniye")
        print(f"  Son Konum: {result['final_position']}")

if __name__ == "__main__":
    main()