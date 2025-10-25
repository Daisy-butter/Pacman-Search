# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()


    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches thepy
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    from util import Stack
    # 定义栈
    fringe = Stack()
    fringe.push(problem.getStartState())
    # 存储已经访问过的节点
    already_visited = []
    # 存储路径
    path = []
    # 存储路径到当前节点
    pathtonow = Stack()
    current_state = fringe.pop()
    while not problem.isGoalState(current_state):
        # 已经访问过的节点不再加入到搜素边缘中
        if current_state not in already_visited:
            already_visited.append(current_state)
            successors = problem.getSuccessors(current_state)
            # 遍历当前节点的所有后继节点
            for nextsuccessors, action, stepCost in successors :
                if nextsuccessors not in already_visited:
                    fringe.push(nextsuccessors)
                    nextpath = path + [action]
                    # 存储到生成节点的路径
                    pathtonow.push(nextpath)
        # 取出下一个节点
        current_state = fringe.pop()
        # 取出到当前节点的路径
        path = pathtonow.pop()
    return path

    # util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    fringe = Queue()
    fringe.push(problem.getStartState())
    already_visited = []
    path = []
    pathtonow = Queue()
    current_state = fringe.pop()
    while not problem.isGoalState(current_state):
        if current_state not in already_visited:
            already_visited.append(current_state)
            successors = problem.getSuccessors(current_state)
            for nextsuccessors, action, stepCost in successors:
                if nextsuccessors not in already_visited:
                    fringe.push(nextsuccessors)
                    nextpath = path + [action]
                    pathtonow.push(nextpath)
        current_state = fringe.pop()
        path = pathtonow.pop()
    return path

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    fringe = PriorityQueue()
    fringe.push(problem.getStartState(), 0)
    already_visited = []
    path = []
    pathtonow = PriorityQueue()
    current_state = fringe.pop()
    while not problem.isGoalState(current_state):
        if current_state not in already_visited:
            already_visited.append(current_state)
            successors = problem.getSuccessors(current_state)
            for nextsuccessors, action, stepCost in successors:
                if nextsuccessors not in already_visited:
                    nextcost = problem.getCostOfActions(path + [action]) + stepCost
                    fringe.push(nextsuccessors, nextcost)
                    nextpath = path + [action]
                    pathtonow.push(nextpath, nextcost)
        current_state = fringe.pop()
        path = pathtonow.pop()
    return path
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    fringe_and_path = PriorityQueue()
    fringe_and_path.push((problem.getStartState(), []), problem.getCostOfActions([]) + heuristic(problem.getStartState(), problem))
    already_visited = []
    path = []
    current_state = fringe_and_path.pop()
    while not problem.isGoalState(current_state[0]):
        if current_state[0] not in already_visited:
            already_visited.append(current_state[0])
            successors = problem.getSuccessors(current_state[0])
            for nextsuccessors, action, stepCost in successors:
                nextpath = path + [action]
                nextcost = problem.getCostOfActions(nextpath) + heuristic(nextsuccessors, problem)

                if nextsuccessors not in already_visited:
                    fringe_and_path.push((nextsuccessors, nextpath), nextcost)

        current_state = fringe_and_path.pop()
        path = current_state[1]
    return path
    #util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
