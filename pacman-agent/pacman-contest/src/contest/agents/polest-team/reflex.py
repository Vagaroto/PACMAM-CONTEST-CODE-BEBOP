from typing import Tuple, List
from contest.game import Directions

import sys
from pathlib import Path

# Add the directory containing 'contest' to sys.path for absolute imports
script_dir = Path(__file__).parent
contest_src_dir = script_dir.parent.parent.parent # This points to pacman-agent/pacman-contest/src
if str(contest_src_dir) not in sys.path:
    sys.path.append(str(contest_src_dir))

from contest.util import manhattan_distance
from contest.capture_agents import CaptureAgent # Import CaptureAgent for type hinting

def reflex_override(agent_instance: CaptureAgent, gameState, agent_index: int, proposed_action_idx: int) -> Tuple[int, bool]:
    """
    Applies reflex-based override rules to the proposed action.
    Priority: immediate death avoidance > carrying retreat > capsule pursuit > chase > safety STOP.

    Args:
        gameState: The current game state object.
        agent_index: The index of the current agent.
        proposed_action_idx: The action index proposed by the PPO policy (0: Stop, 1: North, 2: South, 3: East, 4: West).

    Returns:
        Tuple[int, bool]:
            - final_action_idx: The chosen action index after applying overrides.
            - overridden: A boolean indicating if an override occurred.
    """
    legal_actions = gameState.get_legal_actions(agent_index)
    # Map numerical action index to game.Directions
    action_map = {
        0: Directions.STOP,
        1: Directions.NORTH,
        2: Directions.SOUTH,
        3: Directions.EAST,
        4: Directions.WEST
    }
    reverse_action_map = {v: k for k, v in action_map.items()}

    proposed_action = action_map.get(proposed_action_idx, Directions.STOP)
    overridden = False
    
    my_state = gameState.get_agent_state(agent_index)
    my_pos = my_state.get_position()
    is_pacman = my_state.is_pacman
    
    # Get general game info using the agent_instance's methods
    my_team_indices = agent_instance.get_team(gameState)
    opponent_indices = agent_instance.get_opponents(gameState)
    
    # Helper to get maze distance (or Manhattan if not available/initialized)
    # For this module, we will assume manhattan_distance for simplicity, but a distancer from CaptureAgent would be better
    def get_distance(p1, p2):
        return manhattan_distance(p1, p2)

    # Helper to find path home for pacman
    def get_path_to_home(current_pos, layout):
        # Home territory is left half for red, right half for blue
        if gameState.is_on_red_team(agent_index): # Red team, home is left
            target_x = layout.width // 2 - 1
        else: # Blue team, home is right
            target_x = layout.width // 2
        
        # Find closest point on home border
        best_home_pos = None
        min_dist_to_home = float('inf')
        for y in range(layout.height):
            if not layout.has_wall(target_x, y):
                dist = get_distance(current_pos, (target_x, y))
                if dist < min_dist_to_home:
                    min_dist_to_home = dist
                    best_home_pos = (target_x, y)
        return best_home_pos

    # Helper to find safest alternative action
    def find_safe_alternative(current_pos, legal_actions_list, opponent_positions, avoid_distance=1):
        safe_actions = []
        for action in legal_actions_list:
            if action == Directions.STOP: # Stopping might be safe, but generally discouraged if dangerous
                continue
            successor_pos = gameState.generate_successor(agent_index, action).get_agent_position(agent_index)
            is_safe = True
            for opp_pos in opponent_positions:
                if opp_pos is not None and get_distance(successor_pos, opp_pos) <= avoid_distance:
                    is_safe = False
                    break
            if is_safe:
                safe_actions.append(action)
        return safe_actions

    # --- 1. Immediate Death Avoidance (Highest Priority) ---
    if is_pacman: # Only pacman can be eaten
        opponent_ghost_positions = []
        for opp_idx in opponent_indices:
            opp_state = gameState.get_agent_state(opp_idx)
            if not opp_state.is_pacman and opp_state.get_position() is not None: # Is a ghost and visible
                # Check if ghost is scared, if so, it's not a threat
                if opp_state.scared_timer == 0:
                    opponent_ghost_positions.append(opp_state.get_position())

        if opponent_ghost_positions:
            for action in legal_actions:
                successor_pos = gameState.generate_successor(agent_index, action).get_agent_position(agent_index)
                for ghost_pos in opponent_ghost_positions:
                    if get_distance(successor_pos, ghost_pos) <= 1: # Moving into a ghost
                        safe_actions = find_safe_alternative(my_pos, legal_actions, opponent_ghost_positions, avoid_distance=1)
                        if safe_actions:
                            # Choose a random safe action, or one that maximizes distance
                            # For simplicity, pick the first safe action
                            final_action_idx = reverse_action_map[safe_actions[0]]
                            overridden = True
                            return final_action_idx, overridden
                        else: # No safe move, might have to take the proposed if it's the only option or random
                            pass # Let lower priority rules or NN decide if no immediate death avoidance possible

    # --- 2. Carrying Retreat (High Priority for Pacman) ---
    if is_pacman and my_state.num_carrying >= 2: # Carrying K=2 or more pellets
        threatening_ghosts = []
        for opp_idx in opponent_indices:
            opp_state = gameState.get_agent_state(opp_idx)
            if not opp_state.is_pacman and opp_state.get_position() is not None and opp_state.scared_timer == 0:
                dist_to_ghost = get_distance(my_pos, opp_state.get_position())
                if dist_to_ghost <= 3:
                    threatening_ghosts.append(opp_state.get_position())
        
        if threatening_ghosts:
            home_pos = get_path_to_home(my_pos, gameState.data.layout)
            if home_pos:
                best_action = None
                min_dist_to_home_after_move = float('inf')
                for action in legal_actions:
                    successor_pos = gameState.generate_successor(agent_index, action).get_agent_position(agent_index)
                    dist_after_move = get_distance(successor_pos, home_pos)
                    if dist_after_move < min_dist_to_home_after_move:
                        min_dist_to_home_after_move = dist_after_move
                        best_action = action
                if best_action:
                    final_action_idx = reverse_action_map[best_action]
                    overridden = True
                    return final_action_idx, overridden

    # --- 3. Chase Enemy Pacman (when Ghost) ---
    if not is_pacman: # If agent is a ghost
        enemy_pacman_positions = []
        for opp_idx in opponent_indices:
            opp_state = gameState.get_agent_state(opp_idx)
            if opp_state.is_pacman and opp_state.get_position() is not None:
                dist_to_enemy_pacman = get_distance(my_pos, opp_state.get_position())
                if dist_to_enemy_pacman <= 5: # Within a radius (e.g., 5)
                    enemy_pacman_positions.append(opp_state.get_position())
        
        if enemy_pacman_positions:
            # Choose action that minimizes distance to the closest enemy pacman
            best_action = None
            min_dist_to_target = float('inf')
            closest_enemy_pacman_pos = min(enemy_pacman_positions, key=lambda p: get_distance(my_pos, p))

            for action in legal_actions:
                successor_pos = gameState.generate_successor(agent_index, action).get_agent_position(agent_index)
                dist_after_move = get_distance(successor_pos, closest_enemy_pacman_pos)
                if dist_after_move < min_dist_to_target:
                    min_dist_to_target = dist_after_move
                    best_action = action
            
            if best_action:
                final_action_idx = reverse_action_map[best_action]
                overridden = True
                return final_action_idx, overridden

    # --- 4. Capsule Pursuit under Threat (for Pacman) ---
    if is_pacman:
        available_capsules = gameState.get_capsules()
        if available_capsules:
            closest_capsule = None
            min_capsule_dist = float('inf')
            for capsule_pos in available_capsules:
                dist = get_distance(my_pos, capsule_pos)
                if dist < min_capsule_dist:
                    min_capsule_dist = dist
                    closest_capsule = capsule_pos
            
            # Check for nearby threatening ghosts
            threatening_ghost_nearby = False
            for opp_idx in opponent_indices:
                opp_state = gameState.get_agent_state(opp_idx)
                if not opp_state.is_pacman and opp_state.get_position() is not None and opp_state.scared_timer == 0:
                    dist_to_ghost = get_distance(my_pos, opp_state.get_position())
                    if dist_to_ghost <= 4: # Ghost is near
                        threatening_ghost_nearby = True
                        break

            if closest_capsule and min_capsule_dist <= 5 and threatening_ghost_nearby: # Capsule close and ghost near
                best_action = None
                min_dist_to_capsule_after_move = float('inf')
                for action in legal_actions:
                    successor_pos = gameState.generate_successor(agent_index, action).get_agent_position(agent_index)
                    dist_after_move = get_distance(successor_pos, closest_capsule)
                    if dist_after_move < min_dist_to_capsule_after_move:
                        min_dist_to_capsule_after_move = dist_after_move
                        best_action = action
                if best_action:
                    final_action_idx = reverse_action_map[best_action]
                    overridden = True
                    return final_action_idx, overridden

    # --- 5. STOP Safety Check (Lowest Priority) ---
    if proposed_action == Directions.STOP:
        # Check if stopping is unsafe (e.g., ghost very close)
        is_unsafe_stop = False
        opponent_ghost_positions = []
        for opp_idx in opponent_indices:
            opp_state = gameState.get_agent_state(opp_idx)
            if not opp_state.is_pacman and opp_state.get_position() is not None and opp_state.scared_timer == 0:
                opponent_ghost_positions.append(opp_state.get_position())
        
        if opponent_ghost_positions:
            for ghost_pos in opponent_ghost_positions:
                if get_distance(my_pos, ghost_pos) <= 2: # Ghost is too close to stop safely
                    is_unsafe_stop = True
                    break
        
        if is_unsafe_stop:
            safe_actions = find_safe_alternative(my_pos, legal_actions, opponent_ghost_positions, avoid_distance=1)
            if safe_actions:
                final_action_idx = reverse_action_map[safe_actions[0]]
                overridden = True
                return final_action_idx, overridden
            elif legal_actions and Directions.STOP in legal_actions: # If no safe alternative, but STOP is legal, then STOP. Otherwise, choose a random legal action.
                # If only option is to stop, and it's unsafe, then just pick a random non-stop action if available
                non_stop_actions = [a for a in legal_actions if a != Directions.STOP]
                if non_stop_actions:
                    final_action_idx = reverse_action_map[non_stop_actions[0]] # Just pick first available
                    overridden = True
                    return final_action_idx, overridden


    # If no override, return the proposed action
    return proposed_action_idx, overridden
