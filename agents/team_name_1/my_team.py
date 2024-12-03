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
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food, and returns to base after collecting 6 pieces of food.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_collected = 0  # Contador de comida recogida
        self.base_x_limit = 0  # Límite para la base (mitad izquierda del mapa)
    
    def register_initial_state(self, game_state):
        # Determinar el límite de la base (mitad izquierda del mapa)
        self.base_x_limit = game_state.data.layout.width // 2
        super().register_initial_state(game_state)

    def choose_action(self, game_state):
        """
        Elige la mejor acción considerando comida, evasión de enemigos y la necesidad de regresar a la base.
        """

        actions = game_state.get_legal_actions(self.index)

        # Si el agente ha recogido 6 piezas de comida, se dirige a la base
        if self.food_collected >= 6:
            return self.move_to_base(actions, game_state)

        # Obtener las posiciones de comida disponibles
        food_list = self.get_food(game_state).as_list()

        # Si no hay comida, elige aleatoriamente
        if not food_list:
            return random.choice(actions)

        # Si el agente se encuentra con un enemigo cerca, alejarse
        enemies_in_range = self.get_enemies_in_range(game_state, 5)

        # Si hay enemigos cerca, alejarse de ellos
        if enemies_in_range:
            return self.avoid_enemy(actions, game_state)

        # Si no hay enemigos y comida está disponible, dirigirse a la comida más cercana
        best_action = self.choose_food(actions, game_state, food_list)

        # Aumentar el contador de comida recogida si el agente ha comido una pieza de comida
        if best_action and self.is_food_collected(game_state, best_action):
            self.food_collected += 1

        return best_action

    def choose_food(self, actions, game_state, food_list):
        """
        Elige la acción que lleve al agente hacia la comida más cercana.
        """
        best_action = None
        min_distance = float('inf')

        my_pos = game_state.get_agent_state(self.index).get_position()

        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()

            for food in food_list:
                dist = self.get_maze_distance(successor_pos, food)
                if dist < min_distance:
                    min_distance = dist
                    best_action = action

        return best_action

    def avoid_enemy(self, actions, game_state):
        """
        Elige la mejor acción para alejarse de los enemigos cercanos.
        """
        best_action = None
        max_dist_to_enemy = float('-inf')

        my_pos = game_state.get_agent_state(self.index).get_position()

        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()

            # Evaluar la distancia mínima a los enemigos
            enemies = self.get_enemies_in_range(game_state, 5)
            min_dist_to_enemy = min([self.get_maze_distance(successor_pos, enemy.get_position()) for enemy in enemies])

            if min_dist_to_enemy > max_dist_to_enemy:
                best_action = action
                max_dist_to_enemy = min_dist_to_enemy

        return best_action

    def move_to_base(self, actions, game_state):
        """
        Mueve al agente hacia la base (mitad izquierda del mapa) después de recoger 6 piezas de comida.
        """
        # Si el agente ya está en la base, reinicia el contador de comida y sigue recogiendo
        if self.is_at_base(game_state):
            self.food_collected = 0  # Resetear la comida recogida
            return self.choose_food(actions, game_state, self.get_food(game_state).as_list())

        # Si no está en la base, moverse hacia la mitad izquierda del mapa (hacia la base)
        best_action = None
        min_dist_to_base = float('inf')

        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()

            base_pos = (self.base_x_limit, successor_pos[1])  # Mantener la misma posición y solo modificar X
            dist_to_base = self.get_maze_distance(successor_pos, base_pos)

            if dist_to_base < min_dist_to_base:
                best_action = action
                min_dist_to_base = dist_to_base

        return best_action
    
    def get_enemies_in_range(self, game_state, range_distance):
        """
        Obtiene los enemigos dentro de un rango de `range_distance` desde la posición del agente.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemies_in_range = [enemy for enemy in enemies if enemy.get_position() is not None and self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), enemy.get_position()) <= range_distance]
        return enemies_in_range
    
    def is_at_base(self, game_state):
        """
        Verifica si el agente está en su base (la mitad izquierda del mapa).
        """
        current_pos = game_state.get_agent_state(self.index).get_position()
        return current_pos[0] < self.base_x_limit  # El agente está en la mitad izquierda del mapa

    def is_food_collected(self, game_state, action):
        """
        Verifica si el agente ha recogido comida después de ejecutar la acción.
        """
        # Obtener la nueva posición después de la acción
        successor = self.get_successor(game_state, action)
        successor_pos = successor.get_agent_state(self.index).get_position()

        # Verificar si la nueva posición tiene comida
        food_list = self.get_food(game_state).as_list()
        return successor_pos in food_list




class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
