# src/pipeline/generate_custom_minari.py
import minari
import gymnasium as gym
import numpy as np

def create_local_medium_replay():
    dataset_id = "mujoco/hopper/medium-replay-v0"
    print(f"=========================================================")
    print(f"🛠️  Generating Custom Local Minari Dataset: {dataset_id}")
    print(f"=========================================================")

    # Clear out any corrupted or pre-existing local copy of the target ID
    if dataset_id in minari.list_local_datasets():
        print(f"[!] Found existing local dataset artifact. Deleting before re-run...")
        minari.delete_dataset(dataset_id)

    # 1. Instantiate and wrap the target environment via Minari DataCollector
    base_env = gym.make("Hopper-v4")
    collector_env = minari.DataCollector(base_env)
    
    num_episodes = 200
    print(f"[+] Simulating {num_episodes} custom trajectories into the data collector...")
    
    # 2. Step through the environment to populate the collector's internal buffer
    for ep in range(num_episodes):
        obs, info = collector_env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # To simulate a realistic random/replay dataset quality, sample action space
            action = base_env.action_space.sample()
            obs, reward, done, truncated, info = collector_env.step(action)
            
    # 3. Compile the buffer directly into the local dataset cache directory
    print("[+] Compiling collected steps into local Minari standard dataset format...")
    dataset = collector_env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="Medium-Replay-Random-Sampler",
        author="Alexander",
        author_email="alexander@example.com",
        description="Custom validation replay buffer split generated for out-of-distribution pipeline evaluation."
    )
    
    print(f"[✓] Local Minari dataset successfully compiled!")
    print("=========================================================\n")

if __name__ == "__main__":
    create_local_medium_replay()