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
    Agente que busca comida y evita a los enemigos cuando están cerca. Después de recoger 6 piezas de comida,
    regresa a su base.
    """

    def register_initial_state(self, game_state):
        """
        Inicializa el agente con las coordenadas de la base y el tamaño del mapa.
        """
        # Obtener el tamaño del mapa usando layout
        self.map_width = game_state.data.layout.width  # Ancho del mapa
        self.map_height = game_state.data.layout.height  # Altura del mapa
        
        # Establecer el límite de la base (mitad del mapa)
        self.base_x_limit = self.map_width // 2  # Definir el límite de la base (mitad izquierda del mapa)

        # Contador de comida
        self.food_collected = 0
        
        super().register_initial_state(game_state)

    def is_at_base(self, game_state):
        """
        Verifica si el agente está en su base (territorio).
        La base está definida como la mitad izquierda del mapa.
        """
        current_pos = game_state.get_agent_state(self.index).get_position()
        return current_pos[0] < self.base_x_limit  # El agente está en la mitad izquierda del mapa

    def get_features(self, game_state, action):
        """
        Calcula las características para la acción seleccionada.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # Penalizar por la comida restante

        # Calcular la distancia a la comida más cercana
        if len(food_list) > 0:  # Esto siempre debería ser verdadero, pero por seguridad
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        return features

    def get_weights(self, game_state, action):
        """
        Define los pesos de las características para la acción.
        """
        return {'successor_score': 100, 'distance_to_food': -1}

    def get_enemies_in_range(self, game_state, range_distance):
        """
        Obtiene los enemigos dentro de un rango de `range_distance` desde la posición del agente.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemies_in_range = [enemy for enemy in enemies if enemy.get_position() is not None and self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), enemy.get_position()) <= range_distance]
        return enemies_in_range

    def choose_action(self, game_state):
        """
        Elige la mejor acción, considerando la comida, evitando enemigos cuando el agente está cerca,
        y regresando a la base después de 6 piezas de comida.
        """
        actions = game_state.get_legal_actions(self.index)

        # Si el agente ya ha comido 6 piezas de comida, regresa a la base
        if self.food_collected >= 6:
            # Verifica si el agente está en su base o no
            if self.is_at_base(game_state):
                # Si está en la base, reinicia el contador de comida
                self.food_collected = 0
                return random.choice(actions)  # Después de regresar a la base, puede elegir cualquier acción
            else:
                # Si no está en la base, dirígete hacia la base
                best_action = None
                min_dist_to_base = float('inf')
                for action in actions:
                    successor = self.get_successor(game_state, action)
                    my_pos = successor.get_agent_state(self.index).get_position()
                    dist_to_base = self.get_maze_distance(my_pos, (self.base_x_limit, my_pos[1]))  # Usamos la posición de la base
                    if dist_to_base < min_dist_to_base:
                        best_action = action
                        min_dist_to_base = dist_to_base
                return best_action  # Dirígete a la base si has comido 6 piezas

        # Obtener los enemigos dentro de un rango de 5 bloques
        enemies_in_range = self.get_enemies_in_range(game_state, 5)

        # Si hay enemigos cerca, alejarse de ellos
        if enemies_in_range:
            best_action = None
            max_dist_to_enemy = float('-inf')  # Buscamos la acción que aleje más del peligro

            for action in actions:
                successor = self.get_successor(game_state, action)
                my_pos = successor.get_agent_state(self.index).get_position()

                # Evaluar la distancia mínima a los enemigos
                min_dist_to_enemy = min([self.get_maze_distance(my_pos, enemy.get_position()) for enemy in enemies_in_range])
                
                if min_dist_to_enemy > max_dist_to_enemy:
                    best_action = action
                    max_dist_to_enemy = min_dist_to_enemy

            return best_action  # Si hay enemigos cerca, alejarse de los enemigos

        # Si no hay enemigos cercanos, proceder con la búsqueda de comida
        best_action = None
        best_value = float('-inf')

        for action in actions:
            features = self.get_features(game_state, action)
            weights = self.get_weights(game_state, action)
            score = features * weights
            if score > best_value:
                best_value = score
                best_action = action

        # Si se ha tomado una acción que lleva al agente hacia la comida, contar la comida
        successor = self.get_successor(game_state, best_action)
        food_list = self.get_food(successor).as_list()
        if len(food_list) < len(self.get_food(game_state).as_list()):  # Comió una pieza de comida
            self.food_collected += 1

        return best_action


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
