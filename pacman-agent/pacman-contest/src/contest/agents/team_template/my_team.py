from contest.capture_agents import CaptureAgent
import random, time, util
from contest.game import Directions
import contest.game as game
from collections import deque

# Team Creation
def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent'):
  """
  Creates a team of two specialized agents that divide strategies.
  """
  return [eval(first)(first_index), eval(second)(second_index)]

# Base Agent with Pathfinding
class PathingAgent(CaptureAgent):
  """
  A base class for agents that use A* pathfinding to navigate.
  It also includes a mechanism to track recent positions to detect when it's stuck.
  """
  def register_initial_state(self, game_state):
    CaptureAgent.register_initial_state(self, game_state)
    # Use a deque to store the last 10 positions to detect stuck situations
    self.recent_positions = deque(maxlen=10)

  def a_star_search(self, game_state, start_pos, goal_pos, time_limit=0.95):
    """
    Performs A* search to find the best action to reach the goal.
    Includes a time limit to prevent long calculations from slowing down the agent.
    """
    start_time = time.time()
    open_set = util.PriorityQueue()
    open_set.push((start_pos, []), self.get_maze_distance(start_pos, goal_pos))
    closed_set = set()
    
    while not open_set.is_empty():
      # Check if the search is taking too long
      if time.time() - start_time > time_limit:
        return None

      current_pos, path = open_set.pop()

      if current_pos == goal_pos:
        # Return the first action in the path, or STOP if no path is needed
        return path[0] if path else Directions.STOP

      if current_pos in closed_set:
        continue
      closed_set.add(current_pos)
      
      # Explore neighbors
      x, y = current_pos
      for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
        dx, dy = game.Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        if not game_state.has_wall(next_x, next_y):
          next_pos = (next_x, next_y)
          if next_pos not in closed_set:
            new_path = path + [action]
            # Cost is path length plus heuristic (maze distance)
            cost = len(new_path) + self.get_maze_distance(next_pos, goal_pos)
            open_set.push((next_pos, new_path), cost)
            
    return None # Return None if no path is found

  def _handle_invaders(self, game_state, my_pos):
    """Chase down invaders, but flee if scared."""
    enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
    invaders = [a for a in enemies if a.is_pacman and a.get_position()]
    
    if not invaders:
      return None

    closest_invader_pos = min([inv.get_position() for inv in invaders], key=lambda pos: self.get_maze_distance(my_pos, pos))

    if game_state.get_agent_state(self.index).scared_timer > 0:
      # Flee from the invader
      legal_actions = game_state.get_legal_actions(self.index)
      legal_actions.remove(Directions.STOP) if Directions.STOP in legal_actions else None
      if not legal_actions: return Directions.STOP
      best_action = max(legal_actions, key=lambda action: self.get_maze_distance(game_state.generate_successor(self.index, action).get_agent_position(self.index), closest_invader_pos))
      return best_action
    else:
      # Hunt the invader
      action = self.a_star_search(game_state, my_pos, closest_invader_pos)
      return action if action else Directions.STOP

# Offensive Agent
class OffensiveAgent(PathingAgent):
  """
  An agent focused on offense that can switch to defense if needed.
  """
  def choose_action(self, game_state):
    my_pos = game_state.get_agent_state(self.index).get_position()
    self.recent_positions.append(my_pos)

    # --- Strategy 1: Evade Ghosts (Highest Priority) ---
    evade_action = self._handle_ghost_evasion(game_state, my_pos)
    if evade_action:
      return evade_action

    # --- Role Switch: If invaders are present, become defensive ---
    invader_action = self._handle_invaders(game_state, my_pos)
    if invader_action:
      return invader_action

    # --- Strategy 2: Return Home to Score ---
    target_pos = self._find_home_target(game_state, my_pos)
    
    # --- Strategy 3: Seek Food ---
    if target_pos is None:
      target_pos = self._find_food_target(game_state, my_pos)
    
    # --- Execute Action ---
    if target_pos:
      action = self.a_star_search(game_state, my_pos, target_pos)
      if action:
        return action
      
    # --- Fallback: If no other action is chosen, take a random legal action ---
    legal_actions = game_state.get_legal_actions(self.index)
    legal_actions.remove(Directions.STOP)
    return random.choice(legal_actions) if legal_actions else Directions.STOP

  def _handle_ghost_evasion(self, game_state, my_pos):
    """If a non-scared ghost is nearby, find the safest escape route."""
    opponents = self.get_opponents(game_state)
    ghosts = [g for i in opponents if (g := game_state.get_agent_state(i)).get_position() and not g.is_pacman and g.scared_timer == 0]
    
    if not ghosts:
      return None

    dist_to_ghost = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])

    if dist_to_ghost <= 5:
      legal_actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP]
      possible_positions = [game_state.generate_successor(self.index, a).get_agent_state(self.index).get_position() for a in legal_actions]
      
      if possible_positions:
        # Find the position that is farthest from the nearest ghost
        closest_ghost_pos = min([g.get_position() for g in ghosts], key=lambda pos: self.get_maze_distance(my_pos, pos))
        safest_pos = max(possible_positions, key=lambda pos: self.get_maze_distance(pos, closest_ghost_pos))
        return self.a_star_search(game_state, my_pos, safest_pos)
        
    return None

  def _find_home_target(self, game_state, my_pos):
    """If carrying food, find the closest border position to return to."""
    if game_state.get_agent_state(self.index).num_carrying >= 1:
      border_x = game_state.data.layout.width // 2 + (-1 if self.red else 1)
      home_positions = [(border_x, y) for y in range(game_state.data.layout.height) if not game_state.has_wall(border_x, y)]
      if home_positions:
        return min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
    return None

  def _find_food_target(self, game_state, my_pos):
    """Find the best food pellet to target, avoiding getting stuck."""
    food_list = self.get_food(game_state).as_list()
    if not food_list:
      return None
      
    # If stuck, target the farthest food pellet to break the loop
    is_stuck = len(self.recent_positions) == self.recent_positions.maxlen and len(set(self.recent_positions)) <= 3
    if is_stuck:
      return max(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
    else:
      return min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))

# Defensive Agent
class DefensiveAgent(PathingAgent):
  """
  A defensive agent that can switch to offense if the coast is clear.
  """
  def choose_action(self, game_state):
    my_pos = game_state.get_agent_state(self.index).get_position()
    self.recent_positions.append(my_pos)
    
    # --- Strategy 1: Handle Invaders ---
    action = self._handle_invaders(game_state, my_pos)
    if action:
      return action

    # --- Role Switch: No invaders, so become offensive ---
    # Evade ghosts first
    evade_action = self._handle_ghost_evasion(game_state, my_pos)
    if evade_action:
        return evade_action
        
    # Then, seek food
    target = self._find_food_target(game_state, my_pos)

    # If no food, patrol
    if target is None:
        target = self._handle_stuck_or_patrol(game_state, my_pos)

    # --- Execute Action ---
    if target:
      action = self.a_star_search(game_state, my_pos, target)
      if action:
        return action
        
    # --- Fallback ---
    legal_actions = game_state.get_legal_actions(self.index)
    legal_actions.remove(Directions.STOP) if Directions.STOP in legal_actions and len(legal_actions) > 1 else None
    return random.choice(legal_actions) if legal_actions else Directions.STOP

  def _handle_ghost_evasion(self, game_state, my_pos):
    """If a non-scared ghost is nearby, find the safest escape route."""
    opponents = self.get_opponents(game_state)
    ghosts = [g for i in opponents if (g := game_state.get_agent_state(i)).get_position() and not g.is_pacman and g.scared_timer == 0]
    
    if not ghosts:
      return None

    dist_to_ghost = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])

    if dist_to_ghost <= 5:
      legal_actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP]
      possible_positions = [game_state.generate_successor(self.index, a).get_agent_state(self.index).get_position() for a in legal_actions]
      
      if possible_positions:
        # Find the position that is farthest from the nearest ghost
        closest_ghost_pos = min([g.get_position() for g in ghosts], key=lambda pos: self.get_maze_distance(my_pos, pos))
        safest_pos = max(possible_positions, key=lambda pos: self.get_maze_distance(pos, closest_ghost_pos))
        return self.a_star_search(game_state, my_pos, safest_pos)
        
    return None
  
  def _find_food_target(self, game_state, my_pos):
    """Find the best food pellet to target, avoiding getting stuck."""
    food_list = self.get_food(game_state).as_list()
    if not food_list:
      return None
      
    # If stuck, target the farthest food pellet to break the loop
    is_stuck = len(self.recent_positions) == self.recent_positions.maxlen and len(set(self.recent_positions)) <= 3
    if is_stuck:
      return max(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
    else:
      return min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
      
  def _handle_stuck_or_patrol(self, game_state, my_pos):
    """If stuck, move to the border. Otherwise, patrol a strategic central point."""
    is_stuck = len(self.recent_positions) == self.recent_positions.maxlen and len(set(self.recent_positions)) <= 3
    
    if is_stuck:
      # If stuck, move towards the center of the border to reset
      center_x = game_state.data.layout.width // 2 + (-1 if self.red else 1)
      patrol_points = [(center_x, y) for y in range(game_state.data.layout.height) if not game_state.has_wall(center_x, y)]
      return min(patrol_points, key=lambda pos: self.get_maze_distance(my_pos, pos)) if patrol_points else None
    else:
      # Guard the most central food pellet
      food_to_defend = self.get_food_you_are_defending(game_state).as_list()
      if food_to_defend:
        # Find the centermost point on our side of the map to patrol
        center_x = game_state.data.layout.width // 2 - 1 if self.red else game_state.data.layout.width // 2
        best_pos = None
        min_dist_to_center = float('inf')
        
        for y in range(game_state.data.layout.height):
            if not game_state.has_wall(center_x, y):
                pos = (center_x, y)
                dist = abs(y - game_state.data.layout.height // 2)
                if dist < min_dist_to_center:
                    min_dist_to_center = dist
                    best_pos = pos
        return best_pos
    return None
