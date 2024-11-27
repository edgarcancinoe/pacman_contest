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
    """The Manhattan distance heuristic for a PositionSearchProblem"""
    xy1 = game_state.get_agent_position(index)
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='OffensiveAgent', num_training=0):
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

    def __init__(self, index, time_for_computing=.1, bag_capacity=4):
        super().__init__(index, time_for_computing)
        self.start = None
        self.is_red = None
        self.last_position = None
        self.bag_capacity = bag_capacity

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.last_position = self.start
        self.is_red = game_state.is_on_red_team(self.index)
        print('I am agent', self.index, 'and I am on the red team:', self.is_red)
        CaptureAgent.register_initial_state(self, game_state)

    def relevant_information(self, game_state):
        state_info = dict()
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state) if game_state.get_agent_state(i) is not None]

        # List of observable opponent's positions 
        state_info['opponent_pos'] = [enemy.get_position() for enemy in enemies if enemy.get_position() is not None]
        state_info['n_obs_opponents'] = len(state_info['opponent_pos'])
        # List of observable invader's positions
        state_info['invader_pos'] = [enemy.get_position() for enemy in enemies if enemy.is_pacman and enemy.get_position() is not None]
        state_info['n_obs_invaders'] = len(state_info['invader_pos'])
        # How many food pellets the agent is carrying in the given state
        state_info['num_food_carrying'] = game_state.get_agent_state(self.index).num_carrying

        # Get current position
        state_info['my_pos'] = game_state.get_agent_state(self.index).get_position()

        # True-False knowledge
        state_info['reborn'] = state_info['my_pos'] == self.start and self.get_maze_distance(state_info['my_pos'], self.last_position) > 1
        state_info['opponents_on_sight'] = state_info['n_obs_opponents'] > 0
        state_info['invaders_on_sight'] = state_info['n_obs_invaders'] > 0
        
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

    def evaluate(self, game_state, action):
        """
        ...
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

class OffensiveAgent(PacmanAgent):
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)

        self.a_star_engine = AStarEngine(index, manhattan_heuristic)

    def a_star_step(self, game_state):
        # Check if the agent should be looking for food or returning to base
        if self.state_knowledge['num_food_carrying'] >= self.bag_capacity:
            # Return to base side
            base_x = game_state.get_walls().width // 2 
            base_x = base_x if not self.is_red else base_x - 1
            
            # All x,y where x is base_x and y is not a wall
            base_point = [(base_x, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(base_x, y)]
            # Get the one with minimum distance
            base_point = min(base_point, key=lambda x: self.get_maze_distance(self.state_knowledge['my_pos'], x))
            return self.a_star_engine.get_next_action(game_state, base_point, reset=True)
        
        # If we dont have a path or we have been reborn, create one to the nearest food
        elif self.state_knowledge['reborn'] or not self.a_star_engine.has_path():
            food_list = game_state.get_red_food().as_list() if not self.is_red else game_state.get_blue_food().as_list()
            # Is there still food in the maze?
            if len(food_list) == 0:
                return self.a_star_engine.get_next_action(game_state, self.start, reset=True)
            else:
                # Obtain nearest food
                my_pos = self.state_knowledge['my_pos']
                # Minimun distance to a food pellet
                min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                # List of food pellets at minimum distance and randomly choose one
                nearest_points = [food for food in food_list if self.get_maze_distance(my_pos, food) == min_distance]
                nearest_point = random.choice(nearest_points)
                return self.a_star_engine.get_next_action(game_state, nearest_point, reset=True)
        else:
            return self.a_star_engine.get_next_action()
        
    def expectiminimax(self, game_state):
        return 0
    
    def choose_strategy(self):
        """Choose Strategy depending on the observed state
        
        CASE 1 - No opponents on sight: A* to the nearest food
        CASE 2 - Opponents on sight: ExpectiMiniMax
        """
        if self.state_knowledge['opponents_on_sight']:
            return 'a_star'
            return 'expectiminimax'
        else:
            return 'a_star'

    def choose_action(self, game_state):
        # Call super() to update the state_knowledge
        super().choose_action(game_state)
        machine_state = self.choose_strategy()
        if machine_state == 'a_star':
            return self.a_star_step(game_state)
        elif machine_state == 'expectiminimax':
            return self.expectiminimax(game_state)

    def get_features(self, game_state, is_red):
        features = util.Counter()
        self.red = is_red
        food_list = self.get_food(game_state).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = game_state.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self):
        return {'successor_score': 100, 'distance_to_food': -1}

class DefensiveAgent(PacmanAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def a_star(self, game_state):
        return 0
    
    def expectiminimax(self, game_state):
        return 0
    
    def choose_strategy(self):
        """Choose Strategy depending on the observed state
        
        CASE 1 - No opponents on sight: A* to the biggest cluster of food
        CASE 2 - Opponents on sight: ExpectiMiniMax
        """
        if self.state_knowledge['opponents_on_sight']:
            return 'a_star'
            return 'expectiminimax'
        else:
            return 'a_star'
    
    def choose_action(self, game_state):
        # Call super() to update the state_knowledge
        super().choose_action(game_state)
        machine_state = self.choose_strategy()
        if machine_state == 'a_star':
            return self.a_star(game_state)
        elif machine_state == 'expectiminimax':
            return self.expectiminimax(game_state)
    
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

class ExpectiMinimaxPlanner():
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def evaluation_function(self, game_state):
        return 0
    
    def game_terminal(self, game_state, current_depth):
        return self.max_depth == current_depth or game_state.is_over()
    
    def get_new_state(self, game_state, action, index):
        successor = game_state.generate_successor(index, action)
        pos = successor.get_agent_state(index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(index, action)
        else:
            return successor
    
    def max_value(self, game_state, current_depth, alpha, beta):
        # Check for terminal state, return V, None
        if self.game_terminal(game_state, current_depth):
            return self.evaluation_function(game_state), None
        v = float('-inf')
        best_action = None

        for action in game_state.get_legal_actions(self.index):
            successor = self.get_new_state(game_state, action, self.index)
            # v2, _ = self.min_value(successor, current_depth + 1, alpha, beta)
            # if v2 > v:
            #     v = v2
            #     best_action = action
            # if v > beta:
            #     return v, best_action
            # alpha = max(alpha, v)


    def get_action(self, game_state):
        return self.max

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

    def __init__(self, game_state, cost_fn = lambda x: 1, goal=(1, 1), index=0):
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
        return state.get_agent_position(self.index) == self.goal

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
        import time
        start_time = time.time()
        print(f'Finding path from {game_state.get_agent_state(self.index).get_position()} to {goal} (manh_d = {manhattan_heuristic(game_state, problem, self.index)})')
        actions = self.a_star_search(problem)
        print('Time elapsed:', time.time() - start_time)
        return actions
    
    def has_path(self):
        return len(self.current_path) > 0
    
    def get_next_action(self, game_state = None, goal = None, reset=False):
        if reset or len(self.current_path) == 0:
            self.current_path = self.find_path(game_state, goal)
        if len(self.current_path) == 0:
            return Directions.STOP
        return self.current_path.pop(0)
    