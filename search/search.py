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

def depthFirstSearch(problem):
	"""
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
	"""
    
	"""
    Notes:
    The start coordinate is (5,5).
    The start is not a goal.
    Start's successors are (5,4) and (4,5); this returns a list of length 2;
		each list tuple hold position, direction to go, and stepcost
	"""
	
	#Pseduocode for DFS: https://www.hackerearth.com/practice/algorithms/graphs/depth-first-search/tutorial/
	#More pseudocode that explains where to mark node as visited: https://www.tutorialspoint.com/data_structures_algorithms/depth_first_traversal.htm
	
	#visitedPos is a set containing all the positions on the board that have been visited
	visitedPos = set()
	#create stack that will hold the position we want to go to, and a list of the directions we have to take to get to that position (stores it as a tuple)
	stack = util.Stack()
	stack.push((problem.getStartState(), []))
	
	#DFS Iterative Portion
	while not stack.isEmpty():
		#pop from stack the next position to visit AND directions to that position
		pos, directions = stack.pop()
		#mark that we have visited that position so we don't visit it again
		visitedPos.add(pos)
		#check if we have reached the goal state, return directions to the goal state when we get to it
		if problem.isGoalState(pos):
			return directions
		#push all neighbors (and directions to neighbors) of the position (that are not visited) onto the stack
		for x in problem.getSuccessors(pos):
			if x[0] not in visitedPos:
				stack.push((x[0], directions + [x[1]]))

def breadthFirstSearch(problem):
	"""Search the shallowest nodes in the search tree first."""
	'''
	BFS pseudocode (from COP3530 lecture)
	Take an arbitrary start vertex, mark it identified (color it gray), and place it in a queue.
	while the queue is not empty
		Take a vertex, u, out of the queue and visit u.
		for all vertices, v, adjacent to this vertex, u 
			if v has not been identified or visited
				Mark it identified (color it grey)
				Insert vertex v into the queue.
			We are now finished visiting u (color it purple).
	'''
	
	#visitedPos is a set containing all the positions on the board that have been visited
	visitedPos = set()
	#identifiedPos is a set containing all the positions on the board that have been identified
	identifiedPos = set()
	#create queue that will hold the position we want to go to, and a list of the directions we have to take to get to that position (stores it as a tuple)
	queue = util.Queue()
	queue.push((problem.getStartState(), []))
	
	identifiedPos.add(problem.getStartState())
	#BFS Iterative Portion
	while not queue.isEmpty():
		pos, directions = queue.pop()
		visitedPos.add(pos)
		if problem.isGoalState(pos):
			return directions
		for x in problem.getSuccessors(pos):
			if (x[0] not in visitedPos) and (x[0] not in identifiedPos):
				identifiedPos.add(x[0])
				queue.push((x[0], directions + [x[1]]))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
