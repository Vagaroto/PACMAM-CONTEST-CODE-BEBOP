from contest.capture_agents import CaptureAgent
import random, time, util
from contest.game import Directions
import contest.game as game
from collections import deque

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent'):
  """
  Creates a team of two specialized agents.
  """
  return [eval(first)(first_index), eval(second)(second_index)]

class PathingAgent(CaptureAgent):
  """
  A base class for agents that use A* pathfinding.
  """
  def register_initial_state(self, game_state):
    CaptureAgent.register_initial_state(self, game_state)
    self.recent_positions = deque(maxlen=10)

  def a_star_search(self, game_state, start_pos, goal_pos, time_limit=0.8):
    start_time = time.time()
    open_set = util.PriorityQueue()
    open_set.push((start_pos, []), self.get_maze_distance(start_pos, goal_pos))
    closed_set = set()
    
    while not open_set.is_empty():
      if time.time() - start_time > time_limit: return None
      current_pos, path = open_set.pop()
      if current_pos == goal_pos: return path[0] if path else Directions.STOP
      if current_pos in closed_set: continue
      closed_set.add(current_pos)
      
      x, y = current_pos
      for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
        dx, dy = game.Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        if not game_state.has_wall(next_x, next_y):
          next_pos = (next_x, next_y)
          if next_pos not in closed_set:
            new_path = path + [action]
            cost = len(new_path) + self.get_maze_distance(next_pos, goal_pos)
            open_set.push((next_pos, new_path), cost)
    return None

class OffensiveAgent(PathingAgent):
  """
  An agent that uses A* to pathfind, with logic to get unstuck.
  """
  def choose_action(self, game_state):
    my_state = game_state.get_agent_state(self.index)
    my_pos = my_state.get_position()
    actions = game_state.get_legal_actions(self.index)
    
    self.recent_positions.append(my_pos)

    target_pos = None
    dist_to_ghost = float('inf')
    opponents = self.get_opponents(game_state)
    ghosts = [game_state.get_agent_state(i) for i in opponents if not game_state.get_agent_state(i).is_pacman and game_state.get_agent_state(i).get_position() is not None and game_state.get_agent_state(i).scared_timer == 0]
    if ghosts:
        dist_to_ghost = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])

    if dist_to_ghost <= 5:
        safest_pos = None
        possible_positions = [game_state.generate_successor(self.index, a).get_agent_state(self.index).get_position() for a in actions if a != Directions.STOP]
        if possible_positions:
            safest_pos = max(possible_positions, key=lambda pos: self.get_maze_distance(pos, ghosts[0].get_position()))
        if safest_pos:
            action = self.a_star_search(game_state, my_pos, safest_pos)
            if action: return action

    if my_state.num_carrying >= 1:
      border_x = game_state.data.layout.width // 2
      if self.red: border_x -= 1
      else: border_x += 1
      home_positions = [(border_x, y) for y in range(game_state.data.layout.height) if not game_state.has_wall(border_x, y)]
      if home_positions: target_pos = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))

    if target_pos is None:
      food_list = self.get_food(game_state).as_list()
      if food_list:
        is_stuck = len(self.recent_positions) == self.recent_positions.maxlen and len(set(self.recent_positions)) <= 3
        if is_stuck:
          target_pos = max(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
        else:
          target_pos = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
    
    if target_pos:
      action = self.a_star_search(game_state, my_pos, target_pos)
      if action: return action
      
    best_actions = sorted([a for a in actions if a != Directions.STOP])
    return best_actions[0] if best_actions else Directions.STOP

class DefensiveAgent(PathingAgent):
  """
  A defensive agent with a clear priority system: Hunt > Unstuck > Guard.
  """
  def choose_action(self, game_state):
    my_state = game_state.get_agent_state(self.index)
    my_pos = my_state.get_position()
    actions = game_state.get_legal_actions(self.index)
    actions.remove(Directions.STOP) if Directions.STOP in actions and len(actions) > 1 else None
    
    self.recent_positions.append(my_pos)
    
    target = None
    enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
    invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
    
    if invaders:
      closest_invader_pos = min([inv.get_position() for inv in invaders], key=lambda pos: self.get_maze_distance(my_pos, pos))
      if my_state.scared_timer > 0:
        best_action, max_dist = None, -1
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            dist = self.get_maze_distance(successor.get_agent_state(self.index).get_position(), closest_invader_pos)
            if dist > max_dist: max_dist, best_action = dist, action
        return best_action if best_action else random.choice(actions)
      else:
        target = closest_invader_pos
    
    elif len(self.recent_positions) == self.recent_positions.maxlen and len(set(self.recent_positions)) <= 3:
      center_x = int(game_state.data.layout.width / 2)
      if self.red: center_x -= 1
      else: center_x += 1
      patrol_points = [(center_x, y) for y in range(game_state.data.layout.height) if not game_state.has_wall(center_x, y)]
      if patrol_points:
        target = min(patrol_points, key=lambda pos: self.get_maze_distance(my_pos, pos))

    else:
      food_to_defend = self.get_food_you_are_defending(game_state).as_list()
      if food_to_defend:
        map_center_x = int(game_state.data.layout.width / 2)
        map_center_y = int(game_state.data.layout.height / 2)
        
        while game_state.has_wall(map_center_x, map_center_y) and map_center_y < game_state.data.layout.height -1: map_center_y += 1
        while game_state.has_wall(map_center_x, map_center_y) and map_center_y > 0: map_center_y -= 1
        
        center_pos = (map_center_x, map_center_y)
        target = min(food_to_defend, key=lambda food: self.get_maze_distance(food, center_pos))

    if target:
      action = self.a_star_search(game_state, my_pos, target)
      if action: return action
      
    return random.choice(actions) if actions else Directions.STOP
