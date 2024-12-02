
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
from queue import PriorityQueue

def a_star_search(game_state, start, goal, walls):
    """
    Realiza una búsqueda A* para encontrar el camino óptimo desde `start` a `goal`.

    :param game_state: El estado actual del juego.
    :param start: La posición inicial (tupla: (x, y)).
    :param goal: La posición objetivo (tupla: (x, y)).
    :param walls: Mapa de paredes del juego.
    :return: Lista de acciones para llegar al objetivo.
    """
    # Inicialización
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        _, current = frontier.get()

        # Terminar si llegamos al objetivo
        if current == goal:
            break

        for direction, (dx, dy) in [('North', (0, 1)), ('South', (0, -1)), ('East', (1, 0)), ('West', (-1, 0))]:
            next_pos = (current[0] + dx, current[1] + dy)
            
            # Saltar posiciones inválidas
            if walls[next_pos[0]][next_pos[1]]:
                continue

            # Calcular el costo del movimiento
            new_cost = cost_so_far[current] + 1
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + manhattan_distance(next_pos, goal)
                frontier.put((priority, next_pos))
                came_from[next_pos] = (current, direction)

    # Reconstruir la ruta
    path = []
    current = goal
    while current != start:
        if current not in came_from:
            return []  # Si no hay ruta posible
        prev, action = came_from[current]
        path.append(action)
        current = prev

    path.reverse()  # Invertir la ruta
    return path

def manhattan_distance(pos1, pos2):
    """
    Calcula la distancia de Manhattan entre dos posiciones.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

################################################################

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
    
    def get_astar_action(self, game_state, goal):
        """
        Calcula la acción usando A* para moverse hacia el `goal`.
        """
        start = game_state.get_agent_position(self.index)
        walls = game_state.get_walls()
        path = a_star_search(game_state, start, goal, walls)
        if path:
            return path[0]  # Regresa la primera acción de la ruta
        else:
            return Directions.STOP


class OffensiveReflexAgent(ReflexCaptureAgent):
    def choose_action(self, game_state):
        """
        Selecciona una acción usando A* cuando sea posible.
        """
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)

        # Combina comida y cápsulas como posibles objetivos
        targets = capsules + food_list

        if targets:
            my_pos = game_state.get_agent_position(self.index)
            closest_target = min(targets, key=lambda x: self.get_maze_distance(my_pos, x))

            # Usa A* para calcular la acción hacia el objetivo más cercano
            action = self.get_astar_action(game_state, closest_target)
            if action != Directions.STOP:
                return action

        # Si no hay comida ni cápsulas cerca, usa el comportamiento predeterminado
        return super().choose_action(game_state)

    def get_astar_action(self, game_state, goal):
        """
        Calcula la acción usando A* para moverse hacia el `goal`.
        """
        start = game_state.get_agent_position(self.index)
        walls = game_state.get_walls()
        path = a_star_search(game_state, start, goal, walls)
        if path:
            return path[0]  # Devuelve la primera acción del camino
        else:
            return Directions.STOP

    def get_features(self, game_state, action):
        """
        Modifica los features para el comportamiento ofensivo.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()

        features['successor_score'] = -len(food_list)  # Maximiza comida restante

        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        enemy_positions = [e.get_position() for e in enemies if e.get_position() is not None]

        danger_distance = 9999
        for enemy_pos in enemy_positions:
            danger_distance = min(danger_distance, self.get_maze_distance(my_pos, enemy_pos))

        if danger_distance < 4:
            features['food_in_danger'] = 1
        else:
            features['food_in_danger'] = 0

        return features

    def get_weights(self, game_state, action):
        """
        Define los pesos de los features.
        """
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'food_in_danger': 20
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 1. Definir si estamos en defensa (o si el agente es un Pacman)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # 2. Calcular los enemigos en la zona
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            # Distancia a los invasores
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # 3. Penalizar si nos detenemos innecesariamente
        if action == Directions.STOP:
            features['stop'] = 1
        
        # 4. Penalizar si estamos moviéndonos en reversa
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # 5. Penalizar si estamos alejándonos demasiado de un invasor (si lo detectamos)
        if len(invaders) > 0 and features['invader_distance'] > 3:
            features['move_towards_invader'] = 1
        else:
            features['move_towards_invader'] = 0

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2,
            'move_towards_invader': -20
        }
