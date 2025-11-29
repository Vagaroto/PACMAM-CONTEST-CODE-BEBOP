import torch
import numpy as np
from typing import Tuple, Callable

def encode_state_to_tensors(gameState, agent_index: int, current_step: int, get_maze_distance: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes the game state into grid and vector tensors for neural network input.

    Args:
        gameState: The current game state object.
        agent_index: The index of the current agent.

    Returns:
        A tuple containing:
            - grid_tensor (torch.Tensor): Shape [C, H, W] representing grid features.
            - vector_tensor (torch.Tensor): Shape [F] representing scalar features.
    """
    # Grid dimensions
    grid_width = gameState.data.layout.width
    grid_height = gameState.data.layout.height

    # Initialize grid tensor channels
    # Walls, Food, Capsules, Team1 Territory, Team2 Territory, Visible Teammates, Visible Opponents
    num_grid_channels = 7
    grid_tensor = torch.zeros((num_grid_channels, grid_height, grid_width), dtype=torch.float32)

    # 0: Walls
    for x in range(grid_width):
        for y in range(grid_height):
            if gameState.has_wall(x, y):
                grid_tensor[0, y, x] = 1.0

    # Determine team information
    is_red = gameState.is_on_red_team(agent_index)
    my_team_indices = gameState.get_red_team_indices() if is_red else gameState.get_blue_team_indices()
    opponent_team_indices = gameState.get_blue_team_indices() if is_red else gameState.get_red_team_indices()
    
    # 1: Food
    # Use specific food access methods based on team, as gameState.get_food() is not a direct method.
    # The CaptureAgent's get_food() determines which food is "mine" to eat.
    # Here, we directly map it to either red_food or blue_food from the gameState.
    if is_red:
        food_grid = gameState.get_blue_food() # Red team eats blue food
    else:
        food_grid = gameState.get_red_food() # Blue team eats red food

    for x in range(grid_width):
        for y in range(grid_height):
            if food_grid[x][y]:
                grid_tensor[1, y, x] = 1.0

    # 2: Capsules
    # Similar correction for capsules: CaptureAgent.get_capsules() maps to opponent's capsules
    if is_red:
        capsules = gameState.get_blue_capsules()
    else:
        capsules = gameState.get_red_capsules()
    for x, y in capsules:
        grid_tensor[2, y, x] = 1.0

    # 3: Team 1 Territory Mask (Red Team)
    # 4: Team 2 Territory Mask (Blue Team)
    red_territory_x_limit = gameState.data.layout.width // 2 - 1
    for x in range(grid_width):
        for y in range(grid_height):
            if x <= red_territory_x_limit: # Red's side
                grid_tensor[3, y, x] = 1.0
            else: # Blue's side
                grid_tensor[4, y, x] = 1.0

    # Agent positions and states
    my_state = gameState.get_agent_state(agent_index)
    my_pos = my_state.get_position()
    
    # 5: Visible Teammates
    for teammate_idx in my_team_indices:
        if teammate_idx == agent_index:
            continue
        teammate_state = gameState.get_agent_state(teammate_idx)
        if teammate_state.is_pacman != my_state.is_pacman: # Only consider teammates in different role
            teammate_pos = teammate_state.get_position()
            if teammate_pos is not None:
                x, y = int(teammate_pos[0]), int(teammate_pos[1])
                grid_tensor[5, y, x] = 1.0 # Teammate location

    # 6: Visible Opponents
    for opponent_idx in opponent_team_indices:
        opponent_state = gameState.get_agent_state(opponent_idx)
        if opponent_state.is_pacman != my_state.is_pacman: # Only consider opponents in different role
            opponent_pos = opponent_state.get_position()
            if opponent_pos is not None:
                x, y = int(opponent_pos[0]), int(opponent_pos[1])
                grid_tensor[6, y, x] = 1.0 # Opponent location

    # Vector Tensor Features
    vector_features = []

    # 0: Agent role (0 for Ghost, 1 for Pacman)
    vector_features.append(1.0 if my_state.is_pacman else 0.0)

    # 1: Carried food
    vector_features.append(my_state.num_carrying / 10.0) # Normalize by a plausible max (e.g., 10)

    # 2: Remaining team food counts (normalized by total food)
    # Total food is the sum of red and blue food
    total_food_on_map = gameState.get_red_food().count() + gameState.get_blue_food().count()
    
    if total_food_on_map > 0:
        # Remaining food for my team to eat (opponent's food)
        if is_red:
            remaining_my_target_food = gameState.get_blue_food().count()
        else:
            remaining_my_target_food = gameState.get_red_food().count()
        vector_features.append(remaining_my_target_food / total_food_on_map)
    else:
        vector_features.append(0.0)

    # 3: Scared timers for opponents (max scared time is 40 in capture.py)
    # If the agent is a pacman, it cares about scared opponents to chase them.
    # If the agent is a ghost, it cares about scared teammates, but mostly about its own scared timer.
    # Let's focus on visible opponents' scared timers from the current agent's perspective.
    max_scared_time = 40.0 # Standard max scared time in capture.py
    for opponent_idx in opponent_team_indices:
        opponent_state = gameState.get_agent_state(opponent_idx)
        if opponent_state.scared_timer > 0:
            vector_features.append(opponent_state.scared_timer / max_scared_time)
        else:
            vector_features.append(0.0)
    # Ensure a fixed number of scared timer features (e.g., 2 opponents)
    while len(vector_features) < (3 + len(opponent_team_indices)): # 3 for previous, then opponent timers
        vector_features.append(0.0) # Pad with zeros if fewer than 2 opponents

    # 4: Noisy distances to opponents (normalized by grid diagonal)
    max_dist = np.sqrt(grid_width**2 + grid_height**2)
    for opponent_idx in opponent_team_indices:
        opponent_state = gameState.get_agent_state(opponent_idx)
        opponent_pos = opponent_state.get_position()
        if opponent_pos is not None and my_pos is not None:
            dist = get_maze_distance(my_pos, opponent_pos)
            vector_features.append(dist / max_dist)
        else:
            vector_features.append(1.0) # Max distance if not visible or my_pos is None
    # Ensure a fixed number of noisy distance features
    while len(vector_features) < (3 + 2 * len(opponent_team_indices)): # Pad if needed
        vector_features.append(1.0)

    # 5: Normalized step index
    max_steps = 1200 # A typical max game length in capture.py
    vector_features.append(current_step / max_steps)

    # 6: Score difference (normalized by max possible score)
    # Max possible score is roughly total food on one side.
    # Let's use a heuristic max score, e.g., 20
    max_score = 20.0 
    score_diff = gameState.get_score() if is_red else -gameState.get_score()
    vector_features.append(score_diff / max_score)

    vector_tensor = torch.tensor(vector_features, dtype=torch.float32)

    return grid_tensor, vector_tensor
