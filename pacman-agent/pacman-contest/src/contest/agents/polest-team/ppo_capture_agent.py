import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import yaml
import time
from pathlib import Path
from collections import deque
from typing import List, Dict, Any, Tuple
import hashlib # Added for layout hashing

from contest.game import Agent, Directions
from contest.util import nearest_point
from contest.capture_agents import CaptureAgent

import sys
from pathlib import Path
import json # Moved json import to prevent pylance issues if not used in some paths

# Add the directory containing this script to sys.path for relative imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# Import custom modules (now as absolute imports within their own directory context)
from state_encoder import encode_state_to_tensors
from ppo_network import PPONetwork
from reflex import reflex_override

# Define action mapping from Directions to integer indices
ACTION_MAP = {
    Directions.STOP: 0,
    Directions.NORTH: 1,
    Directions.SOUTH: 2,
    Directions.EAST: 3,
    Directions.WEST: 4
}
REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}
NUM_ACTIONS = len(ACTION_MAP)

class PPOCaptureAgent(CaptureAgent):
    """
    A reinforcement learning Pacman Capture-the-Flag agent using PPO.
    It integrates state encoding, a PPO policy/value network, reflex-based
    action overrides, and handles training data logging and checkpoint management.
    """
    def __init__(self, index, time_for_computing=0.1, **kwargs):
        super().__init__(index, time_for_computing)
        
        # Load configuration
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(self.config['agent_settings']['device'])
        self.inference_mode = self.config['agent_settings'].get('inference_mode', False) # Default to False
        
        # Use original relative paths
        self.checkpoint_dir = Path(__file__).parent / self.config['paths']['checkpoint_dir']
        self.trajectory_dir = Path(__file__).parent / self.config['paths']['trajectory_dir']
        
        # Always ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Only create trajectory directory if not in inference mode
        if not self.inference_mode:
            os.makedirs(self.trajectory_dir, exist_ok=True)

        self.episode_buffer: List[Dict[str, Any]] = []
        self.current_episode = 0
        self.last_checkpoint_timestamp = None

        # These will be initialized in registerInitialState once the layout is known
        self.grid_channels = 7 # Walls, Food, Capsules, Team1 Territory, Team2 Territory, Visible Teammates, Visible Opponents
        self.vector_features = 0 # Will be dynamically determined by state_encoder output length
        self.grid_height = 0
        self.grid_width = 0
        self.layout_hash = "" # Will be computed in register_initial_state

        self.model: PPONetwork = None
        # model loading is deferred to register_initial_state now

    def _compute_layout_hash(self, layout):
        """Computes a hash of the layout's walls for unique identification."""
        # Convert the walls grid to a string representation and hash it
        walls_str = str(layout.walls)
        return hashlib.md5(walls_str.encode('utf-8')).hexdigest()

    def load_model_and_checkpoint(self):
        """
        Initializes the model structure and attempts to load the latest checkpoint
        specific to the current map dimensions and layout hash. If none found, initializes from scratch.
        This is called during __init__ and registerInitialState.
        """
        if self.grid_height == 0 or self.grid_width == 0 or not self.layout_hash:
            # Layout dimensions or hash not yet known, defer model initialization
            return

        self.vector_features = 9 # Assuming 2 opponents for fixed vector_features

        # Construct a dimension and hash-specific glob pattern
        dimension_and_hash_tag = f"_H{self.grid_height}_W{self.grid_width}_L{self.layout_hash}"
        
        latest_checkpoint_path = self._get_latest_checkpoint_path(dimension_and_hash_tag)
        
        if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
            # If a dimension and hash-specific checkpoint is found, load it
            if self.model is None or self.last_checkpoint_timestamp is None or os.path.getmtime(latest_checkpoint_path) > self.last_checkpoint_timestamp:
                self.model, _ = PPONetwork.load_checkpoint(latest_checkpoint_path, self.device)
                self.model.eval() # Set to evaluation mode
                self.last_checkpoint_timestamp = os.path.getmtime(latest_checkpoint_path)
                # print(f"Agent {self.index}: Loaded dimension-specific checkpoint from {latest_checkpoint_path}")
                return
        
        # If no dimension and hash-specific checkpoint found or model is None, initialize from scratch
        if self.model is None:
            # print(f"Agent {self.index}: No dimension-specific checkpoint found for ({self.grid_height}, {self.grid_width}, hash={self.layout_hash}). Initializing model from scratch.")
            self.model = PPONetwork(
                grid_channels=self.grid_channels,
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                vector_features=self.vector_features,
                num_actions=NUM_ACTIONS
            ).to(self.device)
            self.model.eval() # Start in eval mode
            self.last_checkpoint_timestamp = None # Ensure no timestamp if starting from scratch


    def _get_latest_checkpoint_path(self, dimension_and_hash_tag: str = ""):
        """
        Helper to find the most recent checkpoint file, optionally with a dimension and hash tag.
        """
        pattern = f"model_checkpoint{dimension_and_hash_tag}.pt"
        checkpoint_path = self.checkpoint_dir / pattern
        return checkpoint_path if checkpoint_path.exists() else None

    def register_initial_state(self, game_state):
        """
        This method handles the initial setup of the agent.
        Resets episode buffer and periodically reloads checkpoint.
        """
        super().register_initial_state(game_state)
        
        # Initialize grid dimensions and layout hash
        self.grid_height = game_state.data.layout.height
        self.grid_width = game_state.data.layout.width
        self.layout_hash = self._compute_layout_hash(game_state.data.layout)
        self.load_model_and_checkpoint() # Initialize model now that dimensions and hash are known

        if not self.inference_mode:
            self.episode_buffer = [] # Reset episode buffer for a new episode
            self.visited_positions = set() # For novelty-based exploration reward
        self.current_episode += 1

        # Periodically reload checkpoint
        if self.current_episode % self.config['agent_settings']['checkpoint_reload_interval'] == 0:
            self.load_model_and_checkpoint() # Reloads only if a newer checkpoint is available

    def get_action(self, game_state):
        """
        Encodes state, forwards through NN, samples action, applies reflex override,
        and logs transition.
        """
        self.observation_history.append(game_state) # CaptureAgent already does this, but good to ensure
        
        # If halfway between positions, just return the first legal action (standard behavior)
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos != nearest_point(my_pos):
            return game_state.get_legal_actions(self.index)[0]

        # Ensure model is initialized before proceeding
        if self.model is None:
            # If model is not initialized, we can't use NN. Fallback to random action.
            # print(f"Agent {self.index}: Model not initialized, taking random action.")
            legal_actions = game_state.get_legal_actions(self.index)
            if Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP) # Avoid stopping randomly
            if not legal_actions:
                legal_actions = [Directions.STOP]
            return random.choice(legal_actions)

        # 1. Encode state
        # Pass the current step count (length of episode buffer)
        current_step_in_episode = len(self.episode_buffer)
        grid_tensor, vector_tensor = encode_state_to_tensors(game_state, self.index, current_step_in_episode, self.get_maze_distance)
        
        # Add batch dimension
        grid_tensor = grid_tensor.unsqueeze(0).to(self.device)
        vector_tensor = vector_tensor.unsqueeze(0).to(self.device)

        # 2. Forward through NN
        self.model.eval() # Ensure eval mode for inference
        with torch.no_grad():
            policy_logits, value = self.model(grid_tensor, vector_tensor)
        
        # Convert policy_logits to probabilities
        action_probs = F.softmax(policy_logits, dim=-1)
        
        # 3. Sample action during training (or take argmax during evaluation)
        # For now, always sample (assuming we are always training during self-play)
        dist = torch.distributions.Categorical(action_probs)
        proposed_action_idx = dist.sample().item()
        
        # 4. Compute logprob and store value
        log_prob = dist.log_prob(torch.tensor(proposed_action_idx, device=self.device)).item()
        value = value.item()

        # Map proposed action index to game.Directions
        proposed_action = REVERSE_ACTION_MAP.get(proposed_action_idx, Directions.STOP)
        
        # Filter legal actions to ensure we only consider valid moves
        legal_game_actions = game_state.get_legal_actions(self.index)
        if proposed_action not in legal_game_actions:
            # If proposed action is illegal, try to find a legal alternative.
            # For simplicity, if illegal, default to STOP if legal, or a random legal action.
            if Directions.STOP in legal_game_actions:
                proposed_action = Directions.STOP
                proposed_action_idx = ACTION_MAP[Directions.STOP]
            elif legal_game_actions:
                proposed_action = random.choice(legal_game_actions)
                proposed_action_idx = ACTION_MAP[proposed_action]
            else: # No legal actions, must STOP (game should be over)
                proposed_action = Directions.STOP
                proposed_action_idx = ACTION_MAP[Directions.STOP]

        # 5. Apply reflex override
        final_action_idx, overridden = reflex_override(self, game_state, self.index, proposed_action_idx)
        final_action = REVERSE_ACTION_MAP.get(final_action_idx, Directions.STOP)

        # Ensure final_action is legal
        if final_action not in legal_game_actions:
            if Directions.STOP in legal_game_actions:
                final_action = Directions.STOP
                final_action_idx = ACTION_MAP[Directions.STOP]
            elif legal_game_actions:
                final_action = random.choice(legal_game_actions)
                final_action_idx = ACTION_MAP[final_action]
            else: # No legal actions, must STOP (game should be over)
                final_action = Directions.STOP
                final_action_idx = ACTION_MAP[Directions.STOP]


        # 6. Log transition (only if not in inference mode)
        if not self.inference_mode:
            transition = {
                "grid_obs": grid_tensor.squeeze(0).cpu().numpy(), # Remove batch dim, to numpy
                "vector_obs": vector_tensor.squeeze(0).cpu().numpy(),
                "action": final_action_idx, # Store the actual action taken after override
                "log_prob": log_prob,
                "value": value,
                "reward": 0.0, # Placeholder, will be filled by observe_transition
                "overridden": overridden,
                "done": False # Placeholder, will be filled by observe_transition
            }
            self.episode_buffer.append(transition)

        return final_action

    def observe_transition(self, reward: float, terminal: bool):
        """
        Attaches reward and terminal flag to the last transition in the episode buffer.
        """
        if self.inference_mode:
            self.episode_buffer = [] # Clear buffer in inference mode
            return
        
        if self.episode_buffer:
            # --- Combined Exploration Rewards (Optimized) ---
            new_game_state = self.observation_history[-1]
            current_pos = new_game_state.get_agent_state(self.index).get_position()

            # 1. Novelty-Based Reward
            if current_pos not in self.visited_positions:
                reward += self.config['agent_settings'].get('novelty_bonus', 0.02)
                self.visited_positions.add(current_pos)
            
            # 2. Goal-Directed Reward (using Manhattan distance for speed)
            if len(self.observation_history) > 1:
                prev_game_state = self.observation_history[-2]
                prev_pos = prev_game_state.get_agent_state(self.index).get_position()
                
                food_list = self.get_food(new_game_state).as_list()
                
                if food_list:
                    # Use Manhattan distance as a fast heuristic
                    dist_now = min([self.distancer.getDistance(current_pos, food) for food in food_list])
                    dist_prev = min([self.distancer.getDistance(prev_pos, food) for food in food_list])
                    
                    food_bonus = self.config['agent_settings'].get('food_proximity_bonus', 0.03)
                    if dist_now < dist_prev:
                        reward += food_bonus
                    elif dist_now > dist_prev:
                        reward -= food_bonus

            self.episode_buffer[-1]["reward"] = reward
            self.episode_buffer[-1]["done"] = terminal
        
        # We will now call finish_episode_and_save from the final() method
        # so no need to call it here based on terminal flag.

    def finish_episode_and_save(self):
        """
        Writes episode buffer to disk as JSON file.
        """
        if self.inference_mode:
            return

        if not self.episode_buffer:
            return

        # Convert tensors/numpy arrays to lists for JSON serialization
        json_serializable_episode_buffer = []
        for transition in self.episode_buffer:
            serialized_transition = {}
            for k, v in transition.items():
                if isinstance(v, np.ndarray):
                    serialized_transition[k] = v.tolist()
                elif isinstance(v, torch.Tensor):
                    serialized_transition[k] = v.cpu().numpy().tolist()
                else:
                    serialized_transition[k] = v
            json_serializable_episode_buffer.append(serialized_transition)

        # Generate a unique filename
        timestamp = int(time.time())
        # Trajectory filenames for per-layout models should also include dimensions and layout hash
        dimension_and_hash_tag = f"_H{self.grid_height}_W{self.grid_width}_L{self.layout_hash}"
        filename = f"trajectory_agent{self.index}_episode{self.current_episode}_{timestamp}{dimension_and_hash_tag}.json"
        filepath = self.trajectory_dir / filename
        temp_filepath = filepath.with_suffix('.tmp')

        try:
            with open(temp_filepath, 'w') as f:
                json.dump(json_serializable_episode_buffer, f, indent=4)
            os.rename(temp_filepath, filepath) # Atomic operation
            # print(f"Agent {self.index}: Successfully saved JSON trajectory to {filepath} with {len(self.episode_buffer)} steps.")
        except OSError as e:
            print(f"Agent {self.index}: Error saving JSON trajectory to {filepath}: {e}")
        except Exception as e:
            print(f"Agent {self.index}: An unexpected error occurred while saving JSON trajectory to {filepath}: {e}")
            
        self.episode_buffer = [] # Clear buffer after saving

    def final(self, game_state):
        """
        Called at the end of each game to allow for any final processing or saving.
        """
        if self.inference_mode:
            return
            
        self.finish_episode_and_save()
        sys.stdout.flush()
        sys.stderr.flush()
