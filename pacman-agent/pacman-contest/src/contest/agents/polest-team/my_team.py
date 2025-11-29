# my_team.py
# ----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import sys
from pathlib import Path

# Add the directory containing this script to sys.path for relative imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from ppo_capture_agent import PPOCaptureAgent # Now this will work as absolute import within its own directory context

def create_team(first_index, second_index, is_red,
                first_type=PPOCaptureAgent, second_type=PPOCaptureAgent):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    indices.  isRed is True if the red team is being created, and
    False if the blue team is being created.
    """
    return [first_type(first_index), second_type(second_index)]
