# src/pipeline/visualize_agent.py
import argparse
import os
import d3rlpy
import gymnasium as gym
import numpy as np


def visualize(model_path: str, env_name: str, render_mode: str = "human", num_episodes: int = 3, record_video: bool = False, video_folder: str = "../results/videos"):
    """
    Renders a trained d3rlpy policy inside a Gymnasium environment.
    """
    print(f"=========================================================")
    print(f"[+]  Visualizing Policy Agent for: {env_name}")
    print(f"=========================================================")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing at: {model_path}")

    # 1. Load trained policy model payload
    print(f"[+] Loading model payload from: {model_path}")
    algo = d3rlpy.load_learnable(model_path)

    # 2. Determine target Gymnasium environment ID
    if "ant" in env_name.lower():
        gym_id = "Ant-v4"
    elif "hopper" in env_name.lower():
        gym_id = "Hopper-v4"
    elif "walker2d" in env_name.lower():
        gym_id = "Walker2d-v4"
    elif "halfcheetah" in env_name.lower():
        gym_id = "HalfCheetah-v4"
    else:
        gym_id = env_name

    # 3. Instantiate Gymnasium environment with requested rendering mode
    actual_render_mode = "rgb_array" if record_video else render_mode
    print(
        f"[+] Initializing Gymnasium ID: {gym_id} (Render Mode: {actual_render_mode})")

    if gym_id == "Ant-v4":
        # Enable contact forces slicing if trained with 105 dims
        base_env = gym.make(
            gym_id, render_mode=actual_render_mode, use_contact_forces=True)

        class SliceObservationWrapper(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                high = self.env.observation_space.high[:105]
                low = self.env.observation_space.low[:105]
                self.observation_space = gym.spaces.Box(
                    low=low, high=high, dtype=np.float64)

            def observation(self, obs):
                return obs[:105]

        env = SliceObservationWrapper(base_env)
    else:
        env = gym.make(gym_id, render_mode=actual_render_mode)

    # Apply video recording wrapper if requested
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=f"{gym_id.lower()}_evaluation",
            episode_trigger=lambda ep_id: True
        )
        print(
            f"[+] Video recording activated! Output directory: {video_folder}")

    # 4. Simulation Visualization Loop
    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        truncated = False
        episode_return = 0.0
        step_count = 0

        while not (done or truncated):
            action = algo.predict(np.expand_dims(obs, axis=0))[0]
            obs, reward, done, truncated, info = env.step(action)
            episode_return += float(reward)
            step_count += 1

        print(
            f" [+] Episode {ep + 1}/{num_episodes} finished | Total Return: {episode_return:.2f} | Steps: {step_count}")

    env.close()
    print("=========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render or record evaluation episodes of a trained policy.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved .d3 model file")
    parser.add_argument("--environment", type=str, required=True,
                        help="Dataset/Environment identifier string")
    parser.add_argument("--mode", type=str, choices=["human", "rgb_array"], default="human",
                        help="Render mode (human for GUI window, rgb_array for background rendering)")
    parser.add_argument("--record", action="store_true",
                        help="Record video (.mp4) instead of GUI rendering")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of visualization episodes")

    args = parser.parse_args()

    visualize(
        model_path=args.model_path,
        env_name=args.environment,
        render_mode=args.mode,
        num_episodes=args.episodes,
        record_video=args.record
    )
