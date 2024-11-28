# baseline_team.py
# ---------------
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
class MyAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.is_pacman = False
        
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()

        if (self.index % 2 == 0) and my_pos[0] < game_state.data.layout.width // 2:
            self.is_pacman = True  
        else:
            self.is_pacman = False

        if self.is_pacman:
            if self.food_carried >= 4:  # Ejemplo: regresa después de recoger 3 puntos de comida
                return self.return_to_base(game_state, actions)
            else:
                return self.choose_pacman_action(game_state, actions)
        else:
            return self.choose_ghost_action(game_state, actions)
        
    def choose_pacman_action(self, game_state, actions):
        food_list = self.get_food(game_state).as_list()  # Comida en el lado enemigo
        if len(food_list) == 0:
            return random.choice(actions)  # Si no hay comida, toma una acción aleatoria

        my_pos = game_state.get_agent_state(self.index).get_position()
        closest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))

        # Elegir la mejor acción para acercarse a la comida
        best_action = None
        min_dist = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(successor_pos, closest_food)
            if dist < min_dist:
                min_dist = dist
                best_action = action

        # Contabilizar la comida recolectada
        if best_action is not None:
            self.food_carried += 1  # Sumar comida al contador

        return best_action
    
    def return_to_base(self, game_state, actions):
        # Decide la mejor acción para regresar a su lado (lado de su equipo)
        my_pos = game_state.get_agent_state(self.index).get_position()
        start_pos = self.start  # La posición de inicio (en su lado)
        
        best_action = None
        min_dist = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(successor_pos, start_pos)
            if dist < min_dist:
                min_dist = dist
                best_action = action

        # Una vez haya regresado, reseteamos el contador de comida
        self.food_carried = 0

        return best_action
    
    def choose_ghost_action(self, game_state, actions):
        food_list = self.get_food(game_state).as_list()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        pacman_enemies = [e for e in enemies if e.is_pacman]

        if pacman_enemies:
            best_action = None
            min_dist = float('inf')
            for pacman in pacman_enemies:
                pacman_pos = pacman.get_position()
                for action in actions:
                    successor = self.get_successor(game_state, action)
                    successor_pos = successor.get_agent_state(self.index).get_position()
                    dist = self.get_maze_distance(successor_pos, pacman_pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_action = action
            return best_action

        # Si no hay pacman enemigos cerca, simplemente defendemos la zona
        if food_list:
            closest_food = min(food_list, key=lambda food: self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), food))
            best_action = None
            min_dist = float('inf')
            for action in actions:
                successor = self.get_successor(game_state, action)
                successor_pos = successor.get_agent_state(self.index).get_position()
                dist = self.get_maze_distance(successor_pos, closest_food)
                if dist < min_dist:
                    min_dist = dist
                    best_action = action
            return best_action

        return random.choice(actions) 
