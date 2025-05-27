
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from flight_env import FlightEnv

flight_1 = {
    "start": [5.0, 40.0, 8000, 210, 0.0, 68000],
    "goal":  [32.0, 40.0, 8000, 210, 0.0, 68000]
}

env = FlightEnv(flight_1)
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_log")
model.learn(total_timesteps=1000000)

# Evaluation
obs, _ = env.reset()
trajectory = [obs]
rewards = []
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    rewards.append(reward)
    trajectory.append(obs)

trajectory = np.array(trajectory)

def plot_trajectory(trajectory, rewards):
    t = np.arange(len(trajectory))

    plt.figure()
    plt.plot(trajectory[:,0], trajectory[:,1])
    plt.xlabel("Longitude (x)")
    plt.ylabel("Latitude (y)")
    plt.title("Flight Path (x-y)")
    plt.grid()
    plt.savefig("xy_path.png")

    plt.figure()
    plt.plot(t, trajectory[:,2])
    plt.xlabel("Time")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude vs Time")
    plt.grid()
    plt.savefig("altitude_time.png")

    plt.figure()
    plt.plot(t, trajectory[:,3])
    plt.xlabel("Time")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity vs Time")
    plt.grid()
    plt.savefig("velocity_time.png")

    plt.figure()
    plt.plot(t, trajectory[:,5])
    plt.xlabel("Time")
    plt.ylabel("Mass (kg)")
    plt.title("Mass vs Time")
    plt.grid()
    plt.savefig("mass_time.png")

    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward per Step")
    plt.grid()
    plt.savefig("reward_plot.png")

    print(f"Total Reward: {sum(rewards):.2f}")
    print(f"Flight Duration: {len(t)} seconds")
    print(f"Final Mass: {trajectory[-1,5]:.2f} kg (Fuel Consumed: {trajectory[0,5] - trajectory[-1,5]:.2f} kg)")

plot_trajectory(trajectory, rewards)
