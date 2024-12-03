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
    A reflex agent that seeks food with clear decision-making logic
    and evades nearby enemies.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_collected = 0  # Contador de comida recogida

    def choose_action(self, game_state):
        """
        Gestión de acciones basadas en estados:
        1. Si se recogieron 3 piezas de comida, regresa a la base.
        2. Si detecta un enemigo cercano, se aleja.
        3. Si no, busca comida.
        4. Si no hay comida o no hay una opción clara, actúa aleatoriamente.
        """
        # Reinicia el contador si el agente ha sido capturado (vuelve a la posición inicial)
        current_position = game_state.get_agent_position(self.index)
        if current_position == self.start:
            self.food_collected = 0

        actions = game_state.get_legal_actions(self.index)

        # Si hay enemigos cercanos, intenta evadirlos
        enemies = self.get_visible_enemies(game_state)
        if len(enemies) > 0:
            return self.evade_enemy(actions, game_state, enemies)

        # Si recogió 3 piezas de comida, regresa a la base
        if self.food_collected >= 3:
            return self.return_to_base(actions, game_state)

        # Busca comida
        food_list = self.get_food(game_state).as_list()
        if len(food_list) > 0:
            return self.collect_food(actions, food_list, game_state)

        # Actúa aleatoriamente si no hay comida
        return random.choice(actions)

    def collect_food(self, actions, food_list, game_state):
        """
        Mueve al agente hacia la comida más cercana y actualiza el contador de comida.
        """
        my_pos = game_state.get_agent_position(self.index)
        nearest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))

        best_action = None
        min_distance = float('inf')

        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(successor_pos, nearest_food)
            if dist < min_distance:
                min_distance = dist
                best_action = action

        # Verifica si esta acción recoge comida
        if best_action is not None:
            successor = self.get_successor(game_state, best_action)
            if self.get_food(successor).as_list() != self.get_food(game_state).as_list():
                self.food_collected += 1  # Incrementa la comida recogida

        return best_action

    def evade_enemy(self, actions, game_state, enemies):
        """
        Selecciona la acción que maximice la distancia con el enemigo más cercano.
        """
        my_pos = game_state.get_agent_position(self.index)
        enemy_positions = [enemy.get_position() for enemy in enemies if enemy.get_position() is not None]

        best_action = None
        max_min_distance = -1

        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()

            # Calcular la distancia mínima con los enemigos para esta acción
            min_distance_to_enemy = min(
                self.get_maze_distance(successor_pos, enemy_pos) for enemy_pos in enemy_positions
            )

            # Maximizar la distancia mínima a cualquier enemigo
            if min_distance_to_enemy > max_min_distance:
                max_min_distance = min_distance_to_enemy
                best_action = action

        return best_action

    def get_visible_enemies(self, game_state):
        """
        Retorna los enemigos visibles dentro del rango de observación.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_enemies = [enemy for enemy in enemies if enemy.get_position() is not None]
        return visible_enemies

    def return_to_base(self, actions, game_state):
        """
        Mueve al agente hacia la mitad de su lado del mapa y reinicia el contador de comida.
        """
        my_pos = game_state.get_agent_position(self.index)
        boundary_positions = self.get_boundary_positions(game_state)
        nearest_boundary = min(boundary_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))

        best_action = None
        min_distance = float('inf')

        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(successor_pos, nearest_boundary)
            if dist < min_distance:
                min_distance = dist
                best_action = action

        # Reinicia el contador si llega a una posición en la frontera
        if best_action is not None:
            successor = self.get_successor(game_state, best_action)
            if successor.get_agent_position(self.index) in boundary_positions:
                self.food_collected = 0  # Reinicia el contador

        return best_action

    def get_boundary_positions(self, game_state):
        """
        Retorna las posiciones accesibles en la frontera entre ambas mitades del mapa.
        """
        layout_width = game_state.data.layout.width
        layout_height = game_state.data.layout.height
        boundary_x = (layout_width // 2) - 1 if self.red else layout_width // 2

        boundary_positions = []
        for y in range(layout_height):
            if game_state.has_wall(boundary_x, y):
                continue
            boundary_positions.append((boundary_x, y))

        return boundary_positions


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. It now uses noisy distance
    readings to better estimate opponent locations.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.last_known_positions = {}  # Record of opponents' last known positions

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Compute distances to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible_invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(visible_invaders)

        # Track visible invaders
        if len(visible_invaders) > 0:
            invader_distances = [self.get_maze_distance(my_pos, a.get_position()) for a in visible_invaders]
            features['invader_distance'] = min(invader_distances)

        # Handle noisy distances for unobserved opponents
        noisy_distances = game_state.get_agent_distances()
        possible_positions = self.get_possible_positions(game_state, noisy_distances)

        # Prioritize moving towards the most likely opponent positions
        if possible_positions:
            closest_possible_dist = min(self.get_maze_distance(my_pos, pos) for pos in possible_positions)
            features['likely_opponent_distance'] = closest_possible_dist
        else:
            features['likely_opponent_distance'] = 0  # No information available

        # Penalize stopping and reversing
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Adjusts weights to balance defensive priorities.
        """
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'likely_opponent_distance': -5,
            'stop': -100,
            'reverse': -2
        }

    def get_possible_positions(self, game_state, noisy_distances):
        """
        Estimates possible positions for opponents based on noisy distances.
        """
        possible_positions = []
        walls = game_state.get_walls()
        width, height = walls.width, walls.height

        # Iterate over each opponent
        for opponent_index in self.get_opponents(game_state):
            noisy_distance = noisy_distances[opponent_index]
            opponent_positions = []

            # Check all positions on the board
            for x in range(width):
                for y in range(height):
                    if not walls[x][y]:
                        manhattan_distance = abs(x - game_state.get_agent_position(self.index)[0]) + abs(y - game_state.get_agent_position(self.index)[1])
                        if manhattan_distance == noisy_distance:
                            opponent_positions.append((x, y))

            possible_positions.extend(opponent_positions)

        return possible_positions
