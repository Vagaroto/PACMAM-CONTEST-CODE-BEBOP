import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import glob
import time
import yaml
from pathlib import Path 
from typing import Dict, Any, List
import re # Added for regex parsing of filenames

import json # Added for JSON deserialization
import sys
from pathlib import Path

# Add the directory containing this script to sys.path for relative imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from ppo_network import PPONetwork # Now this will work

# Load configuration
config_path = script_dir / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# PPO Trainer Settings
GAMMA = config['ppo_trainer_settings']['gamma']
LAMBDA = config['ppo_trainer_settings']['lambda']
CLIP_EPSILON = config['ppo_trainer_settings']['clip_epsilon']
VALUE_LOSS_COEF = config['ppo_trainer_settings']['value_loss_coef']
ENTROPY_BONUS_COEF = config['ppo_trainer_settings']['entropy_bonus_coef']
LEARNING_RATE = config['ppo_trainer_settings']['learning_rate']
MINIBATCH_SIZE = config['ppo_trainer_settings']['minibatch_size']
EPOCHS_PER_UPDATE = config['ppo_trainer_settings']['epochs_per_update']

CHECKPOINT_DIR = Path(__file__).parent / config['paths']['checkpoint_dir']
TRAJECTORY_DIR = Path(__file__).parent / config['paths']['trajectory_dir']
DEVICE = torch.device(config['agent_settings']['device'])

# Assuming fixed dimensions from state_encoder and ppo_network
# In a more robust system, these might be dynamically determined or passed.
# For contest simplicity, use values consistent with `ppo_capture_agent.py` initialization.
GRID_CHANNELS = 7
VECTOR_FEATURES = 9 # Assuming 2 opponents

def load_trajectories(trajectory_dir: Path, expected_layout_hash: str = None) -> List[Dict[str, Any]]:
    """
    Loads all saved JSON trajectory files and clears them.
    Filters trajectories by expected_layout_hash if provided.
    Dynamically determines grid dimensions and layout hash from the first valid trajectory.
    """
    trajectories_data = []
    # Look for JSON files that contain the layout hash in their filename
    trajectory_files = sorted(trajectory_dir.glob("trajectory_agent*.json"))
    
    for filepath in trajectory_files:
        filename_match = re.search(r"_H(\d+)_W(\d+)_L([0-9a-fA-F]+)\.json$", filepath.name)
        file_layout_hash = None
        if filename_match:
            file_layout_hash = filename_match.group(3)

        if expected_layout_hash and file_layout_hash != expected_layout_hash:
            # print(f"Skipping trajectory {filepath.name} as its layout hash {file_layout_hash} does not match expected {expected_layout_hash}")
            continue # Skip files not matching the expected layout

        try:
            with open(filepath, 'r') as f:
                episode_data = json.load(f)

            if not episode_data:
                print(f"Warning: Episode data for {filepath} is empty. Skipping.")
                os.remove(filepath)
                continue

            grid_obs_list = [torch.tensor(t["grid_obs"], dtype=torch.float32) for t in episode_data]
            vector_obs_list = [torch.tensor(t["vector_obs"], dtype=torch.float32) for t in episode_data]
            actions_list = [torch.tensor(t["action"], dtype=torch.long) for t in episode_data]
            log_probs_list = [torch.tensor(t["log_prob"], dtype=torch.float32) for t in episode_data]
            values_list = [torch.tensor(t["value"], dtype=torch.float32) for t in episode_data]
            rewards_list = [torch.tensor(t["reward"], dtype=torch.float32) for t in episode_data]
            overridden_list = [torch.tensor(t["overridden"], dtype=torch.bool) for t in episode_data]
            dones_list = [torch.tensor(t["done"], dtype=torch.bool) for t in episode_data]
            
            episode_grid_obs = torch.stack(grid_obs_list)
            episode_vector_obs = torch.stack(vector_obs_list)
            episode_actions = torch.stack(actions_list)
            episode_log_probs = torch.stack(log_probs_list)
            episode_values = torch.stack(values_list)
            episode_rewards = torch.stack(rewards_list)
            episode_overridden = torch.stack(overridden_list)
            episode_dones = torch.stack(dones_list)

            current_grid_height = None
            current_grid_width = None
            current_layout_hash = file_layout_hash

            if filename_match:
                current_grid_height = int(filename_match.group(1))
                current_grid_width = int(filename_match.group(2))
            elif episode_grid_obs.ndim == 4: # Fallback if no hash in filename (e.g., old files)
                current_grid_height = episode_grid_obs.shape[2]
                current_grid_width = episode_grid_obs.shape[3]

            if current_grid_height is None or current_grid_width is None or current_layout_hash is None:
                print(f"Warning: Could not determine layout details from filename {filepath.name}. Skipping.")
                os.remove(filepath)
                continue

            trajectories_data.append({
                "grid_obs": episode_grid_obs,
                "vector_obs": episode_vector_obs,
                "actions": episode_actions,
                "log_probs": episode_log_probs,
                "values": episode_values,
                "rewards": episode_rewards,
                "overridden": episode_overridden,
                "dones": episode_dones,
                "grid_height": current_grid_height,
                "grid_width": current_grid_width,
                "layout_hash": current_layout_hash
            })
            os.remove(filepath) # Delete processed file
            print(f"Loaded and deleted {filepath.name}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from trajectory {filepath}: {e}")
            continue
        except Exception as e:
            print(f"Error loading trajectory {filepath}: {e}")
            continue
    return trajectories_data

def compute_advantages_and_returns(trajectories: List[Dict[str, Any]], model: PPONetwork) -> List[Dict[str, Any]]:
    processed_trajectories = []
    model.eval()

    for traj in trajectories:
        rewards = traj["rewards"].to(DEVICE)
        dones = traj["dones"].to(DEVICE)
        grid_obs = traj["grid_obs"].to(DEVICE)
        vector_obs = traj["vector_obs"].to(DEVICE)
        
        advantages = torch.zeros_like(rewards, dtype=torch.float32, device=DEVICE)
        
        last_gae_lam = 0
        num_steps = rewards.shape[0]

        with torch.no_grad():
            # Get actual values from the current model for accurate GAE calculation
            _, current_model_values = model(grid_obs, vector_obs)
            current_model_values = current_model_values.squeeze()

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # If last step, next_value is 0 if done, otherwise value of current state from model as bootstrap
                next_non_terminal = 1.0 - dones[t].float()
                next_value = 0 if dones[t] else current_model_values[t] # Bootstrap with the value of the last state
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_value = 0 if dones[t] else current_model_values[t+1]

            delta = rewards[t] + GAMMA * next_value * next_non_terminal - current_model_values[t]
            advantages[t] = last_gae_lam = delta + GAMMA * LAMBDA * next_non_terminal * last_gae_lam
        
        returns = advantages + current_model_values

        processed_trajectories.append({
            "grid_obs": traj["grid_obs"],
            "vector_obs": traj["vector_obs"],
            "actions": traj["actions"],
            "log_probs": traj["log_probs"],
            "advantages": advantages.cpu(),
            "returns": returns.cpu(),
            "overridden": traj["overridden"],
            "dones": traj["dones"]
        })
    return processed_trajectories

def ppo_update(model: PPONetwork, optimizer: optim.Adam, data: List[Dict[str, Any]]):
    if not data:
        return
    
    all_grid_obs = torch.cat([d["grid_obs"] for d in data]).to(DEVICE)
    all_vector_obs = torch.cat([d["vector_obs"] for d in data]).to(DEVICE)
    all_actions = torch.cat([d["actions"] for d in data]).to(DEVICE)
    all_old_log_probs = torch.cat([d["log_probs"] for d in data]).to(DEVICE)
    all_advantages = torch.cat([d["advantages"] for d in data]).to(DEVICE)
    all_returns = torch.cat([d["returns"] for d in data]).to(DEVICE)
    all_overridden = torch.cat([d["overridden"] for d in data]).to(DEVICE)

    all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

    dataset_size = all_grid_obs.shape[0]
    indices = np.arange(dataset_size)

    for _ in range(EPOCHS_PER_UPDATE):
        np.random.shuffle(indices)
        for start_idx in range(0, dataset_size, MINIBATCH_SIZE):
            end_idx = min(start_idx + MINIBATCH_SIZE, dataset_size)
            batch_indices = indices[start_idx:end_idx]

            batch_grid_obs = all_grid_obs[batch_indices]
            batch_vector_obs = all_vector_obs[batch_indices]
            batch_actions = all_actions[batch_indices]
            batch_old_log_probs = all_old_log_probs[batch_indices]
            batch_advantages = all_advantages[batch_indices]
            batch_returns = all_returns[batch_indices]
            batch_overridden = all_overridden[batch_indices]

            model.train()
            policy_logits, values_pred = model(batch_grid_obs, batch_vector_obs)
            values_pred = values_pred.squeeze()

            dist = torch.distributions.Categorical(logits=policy_logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * batch_advantages
            
            policy_loss = (-torch.min(surr1, surr2) * (~batch_overridden)).mean()
            
            value_loss = F.mse_loss(values_pred, batch_returns)

            loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_BONUS_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_loop():
    print(f"PPO Trainer started on device: {DEVICE}")
    print(f"Looking for trajectories in: {TRAJECTORY_DIR}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Dictionary to hold a trainer instance for each layout hash
    trainer_instances = {}

    while True:
        # Scan for any new trajectory files to discover new layouts
        all_trajectory_files = list(TRAJECTORY_DIR.glob("trajectory_agent*.json"))
        
        # Discover unique layout hashes from filenames
        discovered_layout_hashes = set()
        for filepath in all_trajectory_files:
            filename_match = re.search(r"_L([0-9a-fA-F]+)\.json$", filepath.name)
            if filename_match:
                discovered_layout_hashes.add(filename_match.group(1))

        # Initialize new trainer instances for newly discovered layouts
        for layout_hash in discovered_layout_hashes:
            if layout_hash not in trainer_instances:
                print(f"Discovered new layout hash: {layout_hash}. Initializing a new trainer instance.")
                
                # Load one trajectory to get dimensions
                temp_traj = load_trajectories(TRAJECTORY_DIR, expected_layout_hash=layout_hash)
                if not temp_traj:
                    print(f"Could not load trajectory for new layout hash {layout_hash}, will retry.")
                    continue

                grid_height = temp_traj[0]['grid_height']
                grid_width = temp_traj[0]['grid_width']

                model = PPONetwork(
                    grid_channels=GRID_CHANNELS,
                    grid_height=grid_height,
                    grid_width=grid_width,
                    vector_features=VECTOR_FEATURES
                ).to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

                # Load checkpoint if it exists
                def _get_latest_checkpoint_path_for_layout(height, width, layout_hash):
                    dimension_and_hash_tag = f"_H{height}_W{width}_L{layout_hash}"
                    pattern = f"model_checkpoint*{dimension_and_hash_tag}.pt"
                    checkpoints = sorted(CHECKPOINT_DIR.glob(pattern), key=os.path.getmtime, reverse=True)
                    return checkpoints[0] if checkpoints else None

                latest_checkpoint_path = _get_latest_checkpoint_path_for_layout(grid_height, grid_width, layout_hash)
                if latest_checkpoint_path:
                    model, optimizer_state_dict = PPONetwork.load_checkpoint(latest_checkpoint_path, DEVICE)
                    if optimizer_state_dict:
                        optimizer.load_state_dict(optimizer_state_dict)
                    print(f"Loaded checkpoint for layout {layout_hash} from {latest_checkpoint_path}")
                
                trainer_instances[layout_hash] = {
                    "model": model,
                    "optimizer": optimizer,
                    "grid_height": grid_height,
                    "grid_width": grid_width,
                    "trajectories": temp_traj, # Start with the loaded trajectory
                    "iteration": 0
                }

        if not trainer_instances:
            print("No trajectories found for any layout. Waiting...")
            time.sleep(10)
            continue

        # Iterate through each active trainer instance and perform an update
        for layout_hash, instance in trainer_instances.items():
            instance['iteration'] += 1
            print(f"\n--- Trainer Iteration {instance['iteration']} for Layout {layout_hash} ---")

            # Load any new trajectories for this specific layout
            new_trajectories = load_trajectories(TRAJECTORY_DIR, expected_layout_hash=layout_hash)
            if new_trajectories:
                instance['trajectories'].extend(new_trajectories)
                print(f"Loaded {len(new_trajectories)} new trajectories.")

            if not instance['trajectories']:
                print("No trajectories for this layout in this iteration. Skipping update.")
                continue

            # Compute advantages and update
            processed_data = compute_advantages_and_returns(instance['trajectories'], instance['model'])
            if processed_data:
                ppo_update(instance['model'], instance['optimizer'], processed_data)
                print("PPO update completed.")
            
            # Save checkpoint
            dimension_tag = f"_H{instance['grid_height']}_W{instance['grid_width']}_L{layout_hash}"
            checkpoint_filename = f"model_checkpoint{dimension_tag}.pt"
            checkpoint_path = CHECKPOINT_DIR / checkpoint_filename
            instance['model'].save_checkpoint(checkpoint_path, instance['optimizer'].state_dict())
            print(f"Saved checkpoint to {checkpoint_path}")
            
            instance['trajectories'] = [] # Clear after processing

        time.sleep(15) # Delay before next full scan


if __name__ == "__main__":
    train_loop()
