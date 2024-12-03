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

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def a_star(self, game_state, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(game_state, current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def heuristic(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def get_neighbors(self, game_state, position):
        """
        Devuelve los vecinos accesibles desde una posición.
        """
        x, y = map(int, position)  # Convertir posición a enteros
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Movimientos posibles
            next_pos = (int(x + dx), int(y + dy))  # Asegurarse de que next_pos es una tupla de enteros
            if not game_state.has_wall(next_pos[0], next_pos[1]):  # Comprobar si es válido
                neighbors.append(next_pos)
        return neighbors


    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        # Usamos get_features para obtener las características de cada acción
        values = [self.get_features(game_state, action) for action in actions]

        # Calculamos los pesos usando get_weights y seleccionamos la mejor acción
        best_action = None
        best_value = float('-inf')  # Valor inicial bajo para comparaciones

        for action, value in zip(actions, values):
            weight = self.get_weights(game_state, action)
            score = sum([value[f] * weight.get(f, 0) for f in value])
            if score > best_value:
                best_value = score
                best_action = action

        return best_action


class OffensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)  # Usa el método get_successor de CaptureAgent
        food_list = self.get_food(successor).as_list()

        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            closest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
            path = self.a_star(successor, my_pos, closest_food)
            features['distance_to_food'] = len(path) if path else float('inf')

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = super().get_successor(game_state, action)  # Usa el método get_successor de CaptureAgent

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            closest_invader = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
            invader_pos = closest_invader.get_position()
            path = self.a_star(successor, my_pos, invader_pos)
            features['invader_distance'] = len(path) if path else float('inf')

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
