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

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
    
    def choose_action(self, game_state):
        """
        Selecciona la mejor acción basada en las características y los pesos.
        """
        actions = game_state.get_legal_actions(self.index)

        # Calcula las características y los pesos para cada acción
        values = [self.get_features(game_state, action) for action in actions]
        best_action = None
        best_value = float('-inf')  # Inicializa con un valor bajo

        for action, value in zip(actions, values):
            weights = self.get_weights(game_state, action)
            score = sum([value[f] * weights.get(f, 0) for f in value])
            if score > best_value:
                best_value = score
                best_action = action

        return best_action


    def simulate_position(self, game_state, action):
        """
        Simula la nueva posición del agente tras aplicar una acción.
        """
        x, y = game_state.get_agent_position(self.index)
        dx, dy = Actions.direction_to_vector(action)
        return (int(x + dx), int(y + dy))

    def a_star(self, game_state, start, goal):
        """
        Implementación mejorada de A* que utiliza self.get_maze_distance
        para calcular distancias y evita cálculos en tiempo real.
        """
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(game_state, start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(game_state, current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(game_state, neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def heuristic(self, game_state, position, goal):
        """
        Heurística mejorada para A* que considera enemigos cercanos.
        """
        # Distancia Manhattan hacia el objetivo
        h = abs(position[0] - goal[0]) + abs(position[1] - goal[1])

        # Evitar enemigos cercanos
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        if ghosts:
            # Encuentra el fantasma más cercano
            ghost_distances = [self.get_maze_distance(position, ghost.get_position()) for ghost in ghosts]
            closest_ghost_distance = min(ghost_distances)

            # Penalización si el fantasma está demasiado cerca
            if closest_ghost_distance < 3:  # Por ejemplo, 3 celdas de distancia
                h += (3 - closest_ghost_distance) * 10  # Penalización más alta cuanto más cerca esté

        return h

    def reconstruct_path(self, came_from, current):
        """
        Reconstruye el camino desde la meta al inicio.
        """
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path


    def get_neighbors(self, game_state, position):
        x, y = map(int, position)
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_pos = (int(x + dx), int(y + dy))
            if not game_state.has_wall(next_pos[0], next_pos[1]):
                neighbors.append(next_pos)
        return neighbors

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

            # Calcula el puntaje para la acción basada en las características y los pesos
            score = sum(features[f] * weights.get(f, 0) for f in features)

            if score > best_score:
                best_score = score
                best_action = action

        # Si no se encuentra una mejor acción, elegir una al azar
        if best_action is None:
            best_action = random.choice(actions)

        return best_action

    def get_features(self, game_state, action):
        features = util.Counter()

        # Obtener el sucesor del estado después de tomar la acción
        successor = game_state.generate_successor(self.index, action)
        food_list = self.get_food(successor).as_list()

        # Característica 1: cantidad de comida restante
        features['successor_score'] = -len(food_list)

        # Característica 2: distancia a la comida más cercana
        if food_list:
            my_pos = successor.get_agent_state(self.index).get_position()
            closest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
            path = self.a_star(successor, my_pos, closest_food)
            features['distance_to_food'] = len(path) if path else float('inf')
        else:
            features['distance_to_food'] = 0  # No hay comida, se puede explorar

        # Penalización por quedarse quieto
        if action == Directions.STOP:
            features['stop'] = 1

        # Penalización por moverse en reversa
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        # Asignar pesos a las características
        return {
            'successor_score': 100,  # Más comida es mejor
            'distance_to_food': -1,  # Menor distancia a la comida es mejor
            'stop': -100,  # No queremos detenernos
            'reverse': -2  # Evitar dar marcha atrás
        }

    def a_star(self, game_state, start, goal):
        # Algoritmo A* para encontrar el camino más corto (simplificado)
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
        # Heurística basada en la distancia Manhattan
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, game_state, position):
        # Obtener los vecinos accesibles desde una posición (sin paredes)
        x, y = position
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_pos = (x + dx, y + dy)
            if not game_state.has_wall(next_pos[0], next_pos[1]):
                neighbors.append(next_pos)
        return neighbors




class DefensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()

        # Simular nueva posición tras aplicar la acción
        new_pos = self.simulate_position(game_state, action)

        # Evaluar si el agente está defendiendo
        my_state = game_state.get_agent_state(self.index)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Detectar invasores
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            closest_invader = min(invaders, key=lambda a: self.get_maze_distance(new_pos, a.get_position()))
            invader_pos = closest_invader.get_position()
            path = self.a_star(game_state, new_pos, invader_pos)
            features['invader_distance'] = len(path) if path else float('inf')

        # Penalización por detenerse o retroceder
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

