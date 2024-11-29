
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
from contest.game import Directions
from contest.util import nearest_point
from contest.distance_calculator import Distancer 


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='AgentA', second='AgentB', num_training=0):
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
class Agents(CaptureAgent):
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
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

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
class AgentA(Agents):
    """
    A reflex agent that seeks food and returns to base after collecting 3 pieces of food.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_carried = 0  # Inicializa la cantidad de comida recogida
        self.map_width = None  # Ancho del mapa
        self.base_x_limit = None  # Límite para la base (la mitad del mapa)

    def register_initial_state(self, game_state):
        """
        Registrar el estado inicial del juego para obtener el ancho del mapa
        y definir el límite de la base (la mitad del mapa).
        """
        super().register_initial_state(game_state)
        self.map_width = game_state.data.layout.width
        self.base_x_limit = self.map_width // 2  # La base está en la mitad izquierda del mapa

    def choose_action(self, game_state):
        """
        Elige la mejor acción para buscar comida o regresar a la base.
        """
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Si ha recogido 3 piezas de comida, el agente se mueve hacia la base
        if self.food_carried >= 3:
            return self.move_to_base(actions, game_state)

        # Si el agente está en territorio enemigo (mitad derecha del mapa), evitar enemigos
        if my_pos[0] >= self.base_x_limit:
            safe_actions = self.avoid_enemies(game_state, actions)
            actions = safe_actions if safe_actions else actions

        # El agente sigue buscando comida
        return self.collect_food(game_state, actions)

    def collect_food(self, game_state, actions):
        """
        El agente busca y recoge la comida más cercana, evitando enemigos.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        food_list = self.get_food(game_state).as_list()

        if not food_list:
            return self.move_to_base(actions, game_state)  # Si no hay comida, regresa a la base.

        # Filtrar las acciones seguras antes de buscar comida
        safe_actions = self.avoid_enemies(game_state, actions)
        if not safe_actions:
            return random.choice(actions)  # Si no hay acciones seguras, movimiento aleatorio

        # Evaluar las acciones seguras con la heurística de la comida más cercana
        best_action = None
        min_dist = float('inf')

        for action in safe_actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()

            # Encuentra la comida más cercana
            closest_food = min(food_list, key=lambda food: self.get_maze_distance(successor_pos, food))
            food_distance = self.get_maze_distance(successor_pos, closest_food)

            # Seleccionar la acción que minimiza la distancia a la comida
            if food_distance < min_dist:
                min_dist = food_distance
                best_action = action

        return best_action

    def avoid_enemies(self, game_state, actions):
        """
        Filtra las acciones que mantienen al agente lejos de los enemigos detectados
        solo si está en territorio enemigo (mitad derecha del mapa).
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        enemy_indices = self.get_opponents(game_state)

        # Detectar posiciones de enemigos cercanos
        enemy_positions = []
        for enemy_index in enemy_indices:
            enemy_state = game_state.get_agent_state(enemy_index)
            if enemy_state.is_pacman and enemy_state.get_position():
                enemy_pos = enemy_state.get_position()
                if self.get_maze_distance(my_pos, enemy_pos) <= 5:
                    enemy_positions.append(enemy_pos)

        # Filtrar las acciones inseguras solo si estamos en territorio enemigo
        safe_actions = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()

            # Solo evitamos enemigos si estamos en territorio enemigo
            if my_pos[0] >= self.base_x_limit:  # Si estamos en la mitad derecha
                if all(self.get_maze_distance(successor_pos, enemy) > 5 for enemy in enemy_positions):
                    safe_actions.append(action)
            else:
                # Si estamos en la mitad izquierda, no hacemos nada especial
                safe_actions.append(action)

        return safe_actions

    def move_to_base(self, actions, game_state):
        """
        Mueve al agente hacia la base (mitad izquierda del mapa) después de recoger 3 piezas de comida.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Verifica si el agente ha llegado a su base
        if self.is_at_base(game_state):
            self.food_carried = 0  # Resetear la comida recogida
            return random.choice(actions)  # Regresar a buscar comida si está en la base

        # Mover hacia la base sin preocuparse por el punto más cercano, solo hacia la mitad izquierda
        if my_pos[0] > self.base_x_limit:  # Si el agente está en la mitad derecha, moverse a la izquierda
            if Directions.WEST in actions:
                return Directions.WEST  # Mueve hacia la izquierda
        else:  # Si está en la mitad izquierda o ya en la base, sigue buscando comida
            return self.collect_food(game_state, actions)

        # Si no puede moverse hacia la izquierda, elige una acción aleatoria
        return random.choice(actions)

    def is_at_base(self, game_state):
        """
        Verifica si el agente está en su base (territorio).
        La base está definida como la mitad izquierda del mapa.
        """
        current_pos = game_state.get_agent_state(self.index).get_position()
        return current_pos[0] < self.base_x_limit



        

class AgentB(Agents):
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

