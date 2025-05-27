import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import json

from flight_env import FlightEnv

class TensorboardCallback(BaseCallback):
    """Custom callback for additional logging"""
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_fuel_consumed = []
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # Log additional metrics if available
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if "r" in ep_info and "l" in ep_info:
                    self.episode_rewards.append(ep_info["r"])
                    self.episode_lengths.append(ep_info["l"])

# Define flight plans
flight_1 = {
    "start": [5.0, 40.0, 8000, 210, 0.0, 68000],
    "goal":  [32.0, 40.0, 8000, 210, 0.0, 68000]
}

flight_2 = {
    "start": [30.0, 55.0, 7000, 220, np.radians(40), 67000],
    "goal":  [15.0, 40.0, 9000, 220, 0.0, 67000]
}

flight_3 = {
    "start": [32.0, 45.0, 8000, 210, np.radians(180), 65000],
    "goal":  [5.0, 45.0, 7000, 210, 0.0, 65000]
}

def train_and_evaluate(flight_plan, flight_name, total_timesteps=20000):
    """Train PPO agent and evaluate on given flight plan"""
    print(f"\n{'='*60}")
    print(f"Training for {flight_name}")
    print(f"{'='*60}")
    
    # Create environment
    env = FlightEnv(flight_plan)
    check_env(env, warn=True)
    
    # PPO hyperparameters (tuned for flight control)
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=f"./ppo_logs/{flight_name}",
        verbose=1
    )
    
    # Train the model
    callback = TensorboardCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save the model
    model.save(f"{flight_name}_ppo_model")
    
    # Evaluation
    print(f"\nEvaluating {flight_name}...")
    obs, _ = env.reset()
    trajectory = [obs.copy()]
    rewards = []
    actions = []
    done = False
    step_count = 0
    
    while not done and step_count < 10000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        rewards.append(reward)
        trajectory.append(obs.copy())
        actions.append(action.copy())
        step_count += 1
        
        # Print progress every 1000 steps
        if step_count % 1000 == 0:
            pos_error = info.get('position_error', 0)
            alt_error = info.get('altitude_error', 0)
            print(f"Step {step_count}: Position error: {pos_error:.2f}, Altitude error: {alt_error:.2f}")
    
    trajectory = np.array(trajectory)
    actions = np.array(actions)
    
    # Generate plots
    plot_results(trajectory, actions, rewards, flight_name)
    
    # Print summary
    print(f"\n{flight_name} Summary:")
    print(f"Total Reward: {sum(rewards):.2f}")
    print(f"Flight Duration: {len(trajectory)-1} seconds")
    print(f"Initial Mass: {trajectory[0,5]:.2f} kg")
    print(f"Final Mass: {trajectory[-1,5]:.2f} kg")
    print(f"Fuel Consumed: {trajectory[0,5] - trajectory[-1,5]:.2f} kg")
    print(f"Success: {'Yes' if terminated else 'No (truncated)'}")
    print(f"Final Position Error: {np.sqrt((trajectory[-1,0]-flight_plan['goal'][0])**2 + (trajectory[-1,1]-flight_plan['goal'][1])**2):.2f}°")
    print(f"Final Altitude Error: {abs(trajectory[-1,2]-flight_plan['goal'][2]):.2f} m")
    
    return model, trajectory, actions, rewards

def plot_results(trajectory, actions, rewards, flight_name):
    """Generate all required plots"""
    t = np.arange(len(trajectory))
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. x-y trajectory
    plt.subplot(3, 3, 1)
    plt.plot(trajectory[:,0], trajectory[:,1], 'b-', linewidth=2)
    plt.plot(trajectory[0,0], trajectory[0,1], 'go', markersize=10, label='Start')
    plt.plot(trajectory[-1,0], trajectory[-1,1], 'ro', markersize=10, label='End')
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.title(f"{flight_name}: Flight Path (x-y)")
    plt.grid(True)
    plt.legend()
    
    # 2. Altitude vs Time
    plt.subplot(3, 3, 2)
    plt.plot(t, trajectory[:,2])
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude vs Time")
    plt.grid(True)
    
    # 3. Velocity vs Time
    plt.subplot(3, 3, 3)
    plt.plot(t, trajectory[:,3])
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity vs Time")
    plt.grid(True)
    
    # 4. Mass vs Time
    plt.subplot(3, 3, 4)
    plt.plot(t, trajectory[:,5])
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (kg)")
    plt.title("Mass vs Time")
    plt.grid(True)
    
    # 5. Thrust vs Time
    if len(actions) > 0:
        plt.subplot(3, 3, 5)
        plt.plot(t[:-1], actions[:,0])
        plt.xlabel("Time (s)")
        plt.ylabel("Throttle δ")
        plt.title("Throttle vs Time")
        plt.ylim([0, 1])
        plt.grid(True)
        
        # 6. Bank angle vs Time
        plt.subplot(3, 3, 6)
        plt.plot(t[:-1], np.degrees(actions[:,1]))
        plt.xlabel("Time (s)")
        plt.ylabel("Bank Angle μ (°)")
        plt.title("Bank Angle vs Time")
        plt.grid(True)
        
        # 7. Flight path angle vs Time
        plt.subplot(3, 3, 7)
        plt.plot(t[:-1], np.degrees(actions[:,2]))
        plt.xlabel("Time (s)")
        plt.ylabel("Flight Path Angle γ (°)")
        plt.title("Flight Path Angle vs Time")
        plt.grid(True)
    
    # 8. Heading vs Time
    plt.subplot(3, 3, 8)
    plt.plot(t, np.degrees(trajectory[:,4]))
    plt.xlabel("Time (s)")
    plt.ylabel("Heading ψ (°)")
    plt.title("Heading vs Time")
    plt.grid(True)
    
    # 9. Reward per Step
    plt.subplot(3, 3, 9)
    if len(rewards) > 0:
        plt.plot(rewards)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward per Step")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{flight_name}_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional plot: 3D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2]/1000, 'b-', linewidth=2)
    ax.scatter(trajectory[0,0], trajectory[0,1], trajectory[0,2]/1000, c='g', s=100, label='Start')
    ax.scatter(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2]/1000, c='r', s=100, label='End')
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_zlabel("Altitude (km)")
    ax.set_title(f"{flight_name}: 3D Flight Path")
    ax.legend()
    plt.savefig(f"{flight_name}_3d_trajectory.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Train and evaluate for all three flights
    results = {}
    
    # Flight 1
    model1, traj1, act1, rew1 = train_and_evaluate(flight_1, "Flight_1", total_timesteps=100000)
    results["Flight_1"] = {
        "trajectory": traj1.tolist(),
        "actions": act1.tolist(),
        "rewards": rew1,
        "total_reward": sum(rew1),
        "duration": len(traj1),
        "fuel_consumed": float(traj1[0,5] - traj1[-1,5])
    }
    
    # Flight 2
    model2, traj2, act2, rew2 = train_and_evaluate(flight_2, "Flight_2", total_timesteps=20000)
    results["Flight_2"] = {
        "trajectory": traj2.tolist(),
        "actions": act2.tolist(),
        "rewards": rew2,
        "total_reward": sum(rew2),
        "duration": len(traj2),
        "fuel_consumed": float(traj2[0,5] - traj2[-1,5])
    }
    
    # Flight 3
    model3, traj3, act3, rew3 = train_and_evaluate(flight_3, "Flight_3", total_timesteps=20000)
    results["Flight_3"] = {
        "trajectory": traj3.tolist(),
        "actions": act3.tolist(),
        "rewards": rew3,
        "total_reward": sum(rew3),
        "duration": len(traj3),
        "fuel_consumed": float(traj3[0,5] - traj3[-1,5])
    }
    
    # Save results
    with open("flight_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("ALL FLIGHTS COMPLETED")
    print("="*60)