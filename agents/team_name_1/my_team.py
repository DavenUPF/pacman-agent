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

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.game import Actions
from util import nearest_point


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
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

import heapq
from captureAgents import CaptureAgent
from game import Directions
import util
import random

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
    
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        best_action = None
        best_value = float('-inf')

        for action in actions:
            features = self.get_features(game_state, action)
            weights = self.get_weights(game_state, action)
            score = sum([features[f] * weights.get(f, 0) for f in features])

            if score > best_value:
                best_value = score
                best_action = action

        return best_action

    def get_features(self, game_state, action):
        features = util.Counter()
        # Característica: distancia al objetivo (por ejemplo, comida más cercana)
        features['distance_to_food'] = 1  # Simplificado: solo un ejemplo, aquí calculas distancias o más
        return features

    def get_weights(self, game_state, action):
        return {'distance_to_food': -1}  # Queremos acercarnos a la comida

class OffensiveReflexAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        best_action = None
        best_score = float('-inf')

        for action in actions:
            features = self.get_features(game_state, action)
            weights = self.get_weights(game_state, action)
            score = sum(features[f] * weights.get(f, 0) for f in features)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def get_features(self, game_state, action):
        features = util.Counter()
        # Característica: distancia a la comida más cercana
        food_list = self.get_food(game_state).as_list()
        if food_list:
            my_pos = game_state.get_agent_state(self.index).get_position()
            closest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
            features['distance_to_food'] = self.get_maze_distance(my_pos, closest_food)
        return features

    def get_weights(self, game_state, action):
        return {'distance_to_food': -1}  # Menor distancia a la comida es mejor

class DefensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        new_pos = self.simulate_position(game_state, action)
        features['on_defense'] = 1  # Suponemos que el agente está defendiendo

        # Detectar invasores (enemigos)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if invaders:
            closest_invader = min(invaders, key=lambda a: self.get_maze_distance(new_pos, a.get_position()))
            invader_pos = closest_invader.get_position()
            features['invader_distance'] = self.get_maze_distance(new_pos, invader_pos)

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10}  # Penalización por invasores

    def simulate_position(self, game_state, action):
        x, y = game_state.get_agent_position(self.index)
        dx, dy = Actions.direction_to_vector(action)
        return (int(x + dx), int(y + dy))
