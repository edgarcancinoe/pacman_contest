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


def manhattan_heuristic(game_state, problem, index):
    """The Manhattan distance heuristic for a PositionSearchProblem with multiple goal positions."""
    xy1 = game_state.get_agent_position(index)
    goals = problem.goal  # Assume problem.goal is a list of goal positions
    return min(abs(xy1[0] - g[0]) + abs(xy1[1] - g[1]) for g in goals)

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
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

class PacmanAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1, max_depth=2):
        super().__init__(index, time_for_computing)
        self.max_depth = max_depth
        self.start = None
        self.is_red = None
        self.last_position = None
        self.machine_state = None
        self.initial_food_count = None

    def bag_capacity(self, game_state):
        # Get number of remaining food pellets to carry
        food_remaining =len(game_state.get_red_food().as_list()) if not self.is_red else len(game_state.get_blue_food().as_list())
        if self.initial_food_count is None:
            self.initial_food_count = food_remaining
        remaining_percentage = food_remaining / self.initial_food_count if self.initial_food_count > 0 else 0
        if remaining_percentage > 0.7:
            # If there is more than 70% of initial food, we carry a small percentage (30%)
            dynamic_capacity = int(food_remaining * 0.3)
        elif remaining_percentage > 0.5:
            # If there is more than 50% of initial food, we carry a medium percentage (40%)
            dynamic_capacity = int(food_remaining * 0.4)
        elif remaining_percentage > 0.3:
            # If there is more than 30% of initial food, we carry a medium percentage (60%)
            dynamic_capacity = int(food_remaining * 0.6)
        else:
            # If there is less than 30% of initial food, we carry a high percentage (80%)
            dynamic_capacity = int(food_remaining * 0.8)

        return dynamic_capacity
        
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.last_position = self.start
        self.is_red = game_state.is_on_red_team(self.index)
        # print('I am agent', self.index, 'and I am on the red team:', self.is_red)
        CaptureAgent.register_initial_state(self, game_state)

    def get_opponents_information(self, game_state):
        opponents_info = {}
        observed_opponent_indexes = [i for i in self.get_opponents(game_state) if game_state.get_agent_state(i).get_position() is not None]
        observed_opponent_states = [game_state.get_agent_state(i) for i in observed_opponent_indexes]
        
        observed_opponent_positions = []
        observed_opponent_ispacman = []
        observed_opponent_scared = []
        for enemy in observed_opponent_states:
            position = enemy.get_position()
            if position is not None:
                observed_opponent_positions.append(position)
                observed_opponent_ispacman.append(True) if enemy.is_pacman else observed_opponent_ispacman.append(False)
                if enemy.is_pacman and (enemy.scared_timer > 0):
                    observed_opponent_scared.append(position) 
        
        opponents_info['obs_opponent_indexes'] = observed_opponent_indexes
        opponents_info['obs_opponent_states'] = observed_opponent_states
        opponents_info['obs_opponent_positions'] = observed_opponent_positions
        opponents_info['observed_opponent_ispacman'] = observed_opponent_ispacman
        opponents_info['n_obs_opponents'] = len(observed_opponent_indexes)
        opponents_info['obs_scared_positions'] = observed_opponent_scared

        return opponents_info
    
    def relevant_information(self, game_state):
        
        # Get observable opponents' information and create dict
        state_info = self.get_opponents_information(game_state)

        # How many food pellets the agent is carrying in the given state
        state_info['num_food_carrying'] = game_state.get_agent_state(self.index).num_carrying

        # Get current position
        state_info['my_pos'] = game_state.get_agent_state(self.index).get_position()

        # True-False knowledge
        state_info['scared'] = game_state.get_agent_state(self.index).scared_timer > 0
        state_info['reborn'] = state_info['my_pos'] == self.start and self.get_maze_distance(state_info['my_pos'], self.last_position) > 1
        state_info['opponents_on_sight'] = state_info['n_obs_opponents'] > 0

        # Update last position
        self.last_position = state_info['my_pos']

        return state_info
        
    def choose_action(self, game_state) -> str:
        """
        This method will be super() called by the child class
        """
        # Obtain relevant information about the state
        self.state_knowledge = self.relevant_information(game_state)

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

    def game_terminal(self, game_state, previous_state, current_depth):
        if self.max_depth == current_depth or game_state.is_over():
            return 1
        else:
            return 0

    def evaluate_minimax(self, game_state, previous_state, terminal):
        """
        ...
        """
        if terminal == 2:
            return float("inf")
        else:
            features = self.get_features(game_state, previous_state)
            weights = self.get_weights(game_state)
            return sum(features[feature] * weights.get(feature, 0) for feature in features)
    
    def get_features(self, game_state, previous_state):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()

        # Get observable opponents' information in evaluating state
        opponents_info = self.get_opponents_information(game_state)
        my_pos = game_state.get_agent_position(self.index)

        # Min distance to our house (used if we are a pacman)
        base_x = game_state.get_walls().width // 2
        base_x = base_x if not self.is_red else base_x - 1
        base_point = [(base_x, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(base_x, y)]
        features["min_dist_house"] = (
            min([self.get_maze_distance(my_pos, house) for house in base_point]) if base_point else 0
        )

        # Min distance to opponents
        obs_opponents_positions = opponents_info['obs_opponent_positions']
        distance_to_opponents = [self.get_maze_distance(my_pos, pos) for pos in obs_opponents_positions]
        features["min_dist_opponents"] = min(distance_to_opponents) if distance_to_opponents else 0

        # Min distance to capsules
        caps_list = game_state.get_red_capsules() if not self.is_red else game_state.get_blue_capsules()
        features["min_dist_caps"] = (
            min([self.get_maze_distance(my_pos, cap) for cap in caps_list]) if caps_list else 0
        )

        # Min distance to food
        food_list = game_state.get_red_food().as_list() if not self.is_red else game_state.get_blue_food().as_list()
        
        features["min_dist_food"] = (
            min([self.get_maze_distance(my_pos, food) for food in food_list]) if food_list else 0
        )

        # Safe distance to opponents
        def safe_distance_score(distance, min_safe=3, max_safe=5):
            """
            Returns a score for the distance to an opponent, where:
            - Penalty is high for being too close (distance < min_safe).
            - Penalty is high for being too far (distance > max_safe).
            - Optimal score for being within the safe range (min_safe <= distance <= max_safe).
            """
            if distance < min_safe:
                return -(min_safe - distance) ** 2  # High penalty for being too close
            elif distance > max_safe:
                return -(distance - max_safe) ** 2  # High penalty for being too far
            else:
                return 0  # No penalty for being within the safe range

        features["safe_dist_opponents"] = (
            max(safe_distance_score(dist) for dist in distance_to_opponents) if distance_to_opponents else 0
        )

        obs_opponents_scared_positions = opponents_info['obs_scared_positions']
        features["min_dist_scared_ghost"] = min([self.get_maze_distance(my_pos, ghost) for ghost in obs_opponents_scared_positions]) if obs_opponents_scared_positions else 0

        ## Feature that is 1 if the agent is pacman
        features["is_pacman"] = 1 if game_state.get_agent_state(self.index).is_pacman else 0

        return features
    
class OffensiveAgent(PacmanAgent):

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.machine_state = 'a_star'
        self.a_star_engine = AStarEngine(index, manhattan_heuristic)
        self.expectiminimax_engine = ExpectiMinimaxPlanner( 
                                                           index = index, 
                                                           evaluation_function=self.evaluate_minimax,
                                                           game_terminal_function=self.game_terminal)
        self.name = 'Offensive Agent'

    def game_terminal(self, game_state, previous_state, current_depth):
        if not game_state.get_agent_state(self.index).is_pacman and not self.state_knowledge['scared']:
            prev_opp_info = self.get_opponents_information(previous_state)
            current_opp_info = self.get_opponents_information(game_state)
            # Posicion antigua del pacman
            last_pos = previous_state.get_agent_state(self.index).get_position()
            # vamos a guardar los indices de los pacmans oponentes que estaban a distancia <= 2
            indexes = []
            for ind, pos, pacman in zip(prev_opp_info["obs_opponent_indexes"], prev_opp_info["obs_opponent_positions"], prev_opp_info["observed_opponent_ispacman"]):
                if pacman and self.get_maze_distance(last_pos, pos) <= 2:
                    indexes.append(ind)
            # Si en el estado actual alguno de esos pacmans ya no esta (se lo han comido) devolvemos distinto que 1 y 0
            if any(item in indexes for item in current_opp_info["obs_opponent_indexes"]):
                return 2
        if self.max_depth == current_depth or game_state.is_over():
            return 1
        else:
            return 0

    def get_weights(self, game_state):
        """
        Normally, weights depend on the game state.
        """
        if game_state.get_agent_state(self.index).is_pacman :
            if self.state_knowledge['num_food_carrying'] >= self.bag_capacity(game_state):
                return {"min_dist_opponents" : 30, "min_dist_house" : -40}
            else:
                return {"min_dist_scared_ghost" : -30, "min_dist_opponents" : 25, "min_dist_caps" : -60, "min_dist_food" : -30}
        else:
            if self.state_knowledge['scared']:
                return {"min_dist_opponents" : 20, "min_dist_food" : -10}
            else:
                return {"min_dist_opponents" : -100}
            
    def a_star_step(self, game_state):
        # Check if the agent should be looking for food or returning to base

        # Is there still food in the maze? OR is the agent carrying the maximum amount of food? -> Return to base side
        food_list = game_state.get_red_food().as_list() if not self.is_red else game_state.get_blue_food().as_list()
        if len(food_list) == 0 or self.state_knowledge['num_food_carrying'] >= self.bag_capacity(game_state):
            # Return to base side: All x,y where x is base_x and y is not a wall
            base_x = game_state.get_walls().width // 2 
            base_x = base_x if not self.is_red else base_x - 1
            base_point = [(base_x, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(base_x, y)]
            return self.a_star_engine.get_next_action(game_state, base_point, reset=True)
        
        # If we dont have a path or we have been reborn, create one to the nearest food
        elif self.state_knowledge['reborn'] or not self.a_star_engine.has_path():
            return self.a_star_engine.get_next_action(game_state, food_list, reset=True)
        else:
            return self.a_star_engine.get_next_action()
        
    def expectiminimax_step(self, game_state):
        return self.expectiminimax_engine.get_maximizing_action(game_state, 
                                                                self.state_knowledge['obs_opponent_indexes'])
    
    def choose_strategy(self):
        """Choose Strategy depending on the observed state
        
        CASE 1 - No opponents on sight: A* to the nearest food
        CASE 2 - Opponents on sight: ExpectiMiniMax
        """
        if self.state_knowledge['opponents_on_sight']:
            # if self.machine_state == 'a_star':
                # print(f'{self.name}: Switching to ExpectiMiniMax strategy')
            return 'expectiminimax'
        else:
            if self.machine_state == 'expectiminimax':
                # print(f'{self.name}: Switching to A* strategy')
                self.a_star_engine.reset()
            return 'a_star'

    def choose_action(self, game_state):
        # Call super() to update the state_knowledge
        super().choose_action(game_state)
        self.machine_state = self.choose_strategy()
        if self.machine_state == 'a_star':
            return self.a_star_step(game_state)
        elif self.machine_state == 'expectiminimax':
            return self.expectiminimax_step(game_state)

class DefensiveAgent(PacmanAgent):
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.machine_state = 'a_star'
        self.a_star_engine = AStarEngine(index, manhattan_heuristic)
        self.last_goal_estimation_T = 0 
        self.expectiminimax_engine = ExpectiMinimaxPlanner(
                                                           index = index, 
                                                           evaluation_function=self.evaluate_minimax,
                                                           game_terminal_function=self.game_terminal)
        self.name = 'Defensive Agent'

    def game_terminal(self, game_state, previous_state, current_depth):
        if not game_state.get_agent_state(self.index).is_pacman and not self.state_knowledge['scared']:
            prev_opp_info = self.get_opponents_information(previous_state)
            current_opp_info = self.get_opponents_information(game_state)
            # Pacman's previous position
            last_pos = previous_state.get_agent_state(self.index).get_position()
            # Find opponents at distance <= 2 (reachable by pacman in next time step)
            indexes = []
            for ind, pos, pacman in zip(prev_opp_info["obs_opponent_indexes"], prev_opp_info["obs_opponent_positions"], prev_opp_info["observed_opponent_ispacman"]):
                if pacman and self.get_maze_distance(last_pos, pos) <= 2:
                    indexes.append(ind)
            # Return if any of the previous on-sight pacmans can't be seen anymore (means they have been eaten) because they were at distance <= 2
            if any(item in indexes for item in current_opp_info["obs_opponent_indexes"]):
                return 2
        if self.max_depth == current_depth or game_state.is_over():
            return 1
        else:
            return 0
        
    def get_weights(self, game_state):
        """
        Normally, weights depend on the game state.
        """
        # We are a pacman: Are vulnerable to ghosts
        if game_state.get_agent_state(self.index).is_pacman :
            return {"is_pacman" : -1000}
        else:
            # We are a ghost: We want to chase the pacman unless we are scared
            if self.state_knowledge['scared']:
                return {"safe_dist_opponents" : 40}
            else:
                return {"min_dist_opponents" : -100}
            
    def a_star_step(self, game_state, num_times_a_star):
        noisy_distances = game_state.get_agent_distances()
        opponent_indxs = game_state.get_red_team_indices() if not self.is_red else game_state.get_blue_team_indices()
        opp_distances = [noisy_distances[i] for i in opponent_indxs]
        
        best_positions = []

        # Recorre cada oponente
        # Obtiene la posición de tu Pacman
        my_pos = game_state.get_agent_position(self.index)

        for i, noisy_distance in enumerate(opp_distances):
            opponent_positions = util.Counter()  # Almacena las posiciones posibles y sus probabilidades
            
            # Recorre todas las posiciones posibles del tablero y calcula la probabilidad de que el oponente esté en cada una
            for x in range(game_state.get_walls().width):
                for y in range(game_state.get_walls().height):
                    # Check if there's no wall at the position (x, y)
                    if not game_state.has_wall(x, y):
                        # Calculate true distance and probability
                        true_dist = self.get_maze_distance(my_pos, (x, y))
                        prob = game_state.get_distance_prob(true_dist, noisy_distance)
                        # Update the opponent's position with the computed probability
                        opponent_positions[(x, y)] += prob
                                    
            best_position = max(opponent_positions, key=opponent_positions.get)
            best_positions.append(best_position)

        in_zone = []
        base_x = game_state.get_walls().width // 2 
        for x, y in best_positions:
            if self.is_red and x < base_x:
                in_zone.append((x, y))
            elif not self.is_red and x >= base_x:
                in_zone.append((x, y))
        # print(in_zone)
        if in_zone:
            goal = min(in_zone, key=lambda opp: self.get_maze_distance(my_pos, opp))
        else:
            (bx, by)= min(best_positions, key=lambda opp: self.get_maze_distance(my_pos, opp))
            base_x = base_x if not self.is_red else base_x - 1
            base_points = [(base_x, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(base_x, y)]
            goal = min(base_points, key=lambda opp: self.get_maze_distance((bx, by), opp))

        if num_times_a_star == 0:
            return self.a_star_engine.get_next_action(game_state, [goal], reset=True)
        else:
            return self.a_star_engine.get_next_action(game_state, [goal], reset=False)
        
    def expectiminimax_step(self, game_state):
        return self.expectiminimax_engine.get_maximizing_action(game_state, 
                                                                self.state_knowledge['obs_opponent_indexes'])
    
    def choose_strategy(self):
        """Choose Strategy depending on the observed state
        
        CASE 1 - No opponents on sight: A* to the nearest food
        CASE 2 - Opponents on sight: ExpectiMiniMax
        """
        if self.state_knowledge['opponents_on_sight']:
            # if self.machine_state == 'a_star':
                # print(f'{self.name}: Switching to ExpectiMiniMax strategy')
            return 'expectiminimax'
        else:
            if self.machine_state == 'expectiminimax':
                # print(f'{self.name}: Switching to A* strategy')
                self.a_star_engine.reset()
            return 'a_star'

    def choose_action(self, game_state):
        num_times_a_star = -1
        # Call super() to update the state_knowledge
        super().choose_action(game_state)
        self.machine_state = self.choose_strategy()
        if self.machine_state == 'a_star':
            num_times_a_star = (num_times_a_star + 1) % 3
            return self.a_star_step(game_state, num_times_a_star)
        elif self.machine_state == 'expectiminimax':
            return self.expectiminimax_step(game_state)

########################################################################
#####################   ExpectiMinimax Classes    ######################
########################################################################

class ExpectiMinimaxPlanner():
    def __init__(self, index=None, evaluation_function=None, game_terminal_function=None):
        self.max_agent_index = index
        assert evaluation_function is not None, 'Evaluation function must be provided'
        self.evaluation_function = evaluation_function
        self.game_terminal=game_terminal_function
    
    def get_new_state(self, game_state, action, index):
        successor = game_state.generate_successor(index, action)
        pos = successor.get_agent_state(index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(index, action)
        else:
            return successor
    
    def expect_value(self, current_state, current_depth, alpha, beta, enemies_on_sight, previous_state):
        terminal = self.game_terminal(current_state, previous_state, current_depth)
        if terminal != 0:
            return self.evaluation_function(current_state, previous_state, terminal)
        
        final_value = 0
        remaining_enemies = enemies_on_sight.copy()
        expect_agent_index = remaining_enemies.pop(0)

        legal_actions = current_state.get_legal_actions(expect_agent_index)
        action_probabilities = [1 / len(legal_actions) for _ in range(len(legal_actions))]
        for action, prob in zip(legal_actions, action_probabilities):
            next_state = self.get_new_state(current_state, action, expect_agent_index)
            if len(remaining_enemies) == 0:
                value = self.max_value(next_state, current_depth + 1, alpha, beta, enemies_on_sight, current_state)
            else:
                value = self.expect_value(next_state, current_depth, alpha, beta, remaining_enemies, current_state)
            
            final_value += prob * value
        return final_value

    def max_value(self, current_state, current_depth, alpha, beta, enemies_on_sight, previous_state):
        terminal = self.game_terminal(current_state, previous_state, current_depth)
        if terminal != 0:
            return self.evaluation_function(current_state, previous_state, terminal)
        
        v = float('-inf')
        for action in current_state.get_legal_actions(self.max_agent_index):
            next_state = self.get_new_state(current_state, action, self.max_agent_index)
            v = max(v, self.opponent_action_value(next_state, current_depth, alpha, beta, enemies_on_sight, current_state))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def get_maximizing_action(self, game_state, enemies_on_sight):
        best_value = float('-inf')
        best_action = None
        
        for action in game_state.get_legal_actions(self.max_agent_index):
            next_state = self.get_new_state(game_state, action, self.max_agent_index)
            v = self.opponent_action_value(next_state, 0, float('-inf'), float('inf'), enemies_on_sight, game_state)
            if v > best_value:
                best_value = v
                best_action = action
        return best_action

    def min_value(self, current_state, current_depth, alpha, beta, enemies_on_sight, previous_state):
        terminal = self.game_terminal(current_state, previous_state, current_depth)
        if terminal != 0:
            return self.evaluation_function(current_state, previous_state, terminal)
        v = float('inf')
        remaining_enemies = enemies_on_sight.copy()
        min_agent_index = remaining_enemies.pop(0)
        for action in current_state.get_legal_actions(min_agent_index):
            next_state = self.get_new_state(current_state, action, min_agent_index)
            if len(remaining_enemies) == 0:
                v = min(v, self.max_value(next_state, current_depth + 1, alpha, beta, enemies_on_sight, current_state))
            else:
                v = min(v, self.min_value(next_state, current_depth, alpha, beta, remaining_enemies, current_state))
            
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def opponent_action_value(self, state, current_depth, alpha, beta, enemies_on_sight, previous_state):
        return self.min_value(state, current_depth, alpha, beta, enemies_on_sight, previous_state)

########################################################################
#########################   A STAR CLASSES    ##########################
########################################################################

# We took this class from the previous pacman project implementation by UC Berkeley
class SearchNode: 
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
class PositionSearchProblem():
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.
    """

    def __init__(self, game_state, cost_fn = lambda x: 1, goal= [(0, 0)], index=0):
        """
        Stores the start and goal.

        game_state: A GameState obj (pacman.py)
        cost_fn: A function from a search state (tuple) to a non-negative number
        goal: A position in the game_state
        """
        self.goal = goal
        self.index = index
        self.startState = game_state
        self.cost_fn = cost_fn

    def get_start_state(self):
        return self.startState

    def is_goal_state(self, state):
        return state.get_agent_position(self.index) in self.goal

    def get_successors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

        As noted in search.py:
        For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in state.get_legal_actions(self.index):
            next_state = state.generate_successor(self.index, action)
            cost = self.cost_fn(next_state)
            successors.append( (next_state, action, cost) )

        return successors

class AStarEngine():
    def __init__(self, index, heuristic = lambda a, b, c : 0):
        self.heuristic = heuristic
        self.index = index
        self.current_path = []

    def a_star_search(self, problem):
        """Search the node that has the lowest combined cost and heuristic first."""
        # We define priority queue data structure
        priority_queue = util.PriorityQueue()
        # Dictionary to keep track of visited nodes (and minimum costs as these are not accessible from the priority queue structure)
        found_costs = dict()

        # Initialize search
        initial_state = problem.get_start_state()
        found_costs[initial_state] = 0
        priority_queue.push(SearchNode(None, (initial_state, None, 0)), 0)

        while not priority_queue.is_empty():
            # Expand node
            expanded_node = priority_queue.pop()
            expanded_state = expanded_node.state

            if problem.is_goal_state(expanded_state):
                return expanded_node.get_path()
                    
            # Add nodes to frontier (priority queue)
            successors = problem.get_successors(expanded_state)
            for successor_info in successors:
                new_state, new_action, new_cost = successor_info
                
                # Current cost to reach the successor node
                cost_to_new_state = found_costs[expanded_state] + new_cost
                
                if new_state not in found_costs or cost_to_new_state < found_costs[new_state]:
                    # Heuristic cost to reach the goal from the successor node
                    h_n = self.heuristic(new_state, problem, self.index)
                    f_n = cost_to_new_state + h_n

                    found_costs[new_state] = cost_to_new_state
                    new_node = SearchNode(expanded_node, (new_state, new_action, cost_to_new_state)) 
                    priority_queue.update(new_node, f_n)
        return None

    def find_path(self, game_state, goal):
        problem = PositionSearchProblem(game_state=game_state, goal=goal, index=self.index)
        actions = self.a_star_search(problem)
        return actions
    
    def has_path(self):
        return len(self.current_path) > 0
    
    def get_next_action(self, game_state = None, goal = None, reset=False):
        if reset or len(self.current_path) == 0:
            self.current_path = self.find_path(game_state, goal)
        if len(self.current_path) == 0:
            return Directions.STOP
        return self.current_path.pop(0)
    
    def reset(self):
        self.current_path = []