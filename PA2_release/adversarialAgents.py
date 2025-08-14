"""
CS311 Programming Assignment 2: Adversarial Search

Full Name: AJ Noyes

Brief description of my evaluation function: N/A
"""

import math, random, typing

import util
from game import Agent, Directions
from pacman import GameState



class ReflexAgent(Agent):
    """
    A reflex agent chooses the best action at each choice point by examining its alternatives via a state evaluation
    function.

    The code below is provided as a guide. You are welcome to change it as long as you don't modify the method
    signatures.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState) -> str:
        """Choose the best action according to an evaluation function.

        Review pacman.py for the available methods on GameState.

        Args:
            gameState (GameState): Current game state

        Returns:
            str: Chosen legal action in this state
        """
        # Collect legal moves
        legalMoves = gameState.getLegalActions()

        # Compute the score for the successor states, choosing the highest scoring action
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        # Break ties randomly
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, gameState: GameState, action: str):
        """Compute score for current game state and proposed action"""
        successorGameState = gameState.generatePacmanSuccessor(action)
        return successorGameState.getScore()


def scoreEvaluationFunction(gameState: GameState) -> float:
    """
    Return score of gameState (as shown in Pac-Man GUI)

    This is the default evaluation function for adversarial search agents (not reflex agents)
    """
    return gameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    Abstract Base Class for Minimax, AlphaBeta and Expectimax agents.

    You do not need to modify this class, but it can be a helpful place to add attributes or methods that used by
    all your agents. Do not remove any existing functionality.
    """

    def __init__(self, evalFn=scoreEvaluationFunction, depth=2):
        self.index = 0  # Pac-Man is always agent index 0
        self.evaluationFunction = globals()[evalFn] if isinstance(evalFn, str) else evalFn
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """Minimax Agent"""


    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action from the current gameState.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """

        """
        Some potentially useful methods on GameState (recall Pac-Man always has an agent index of 0, the ghosts >= 1):

        getLegalActions(agentIndex): Returns a list of legal actions for an agent
        generateSuccessor(agentIndex, action): Returns the successor game state after an agent takes an action
        getNumAgents(): Return the total number of agents in the game
        getScore(): Return the score corresponding to the current state of the game
        isWin(): Return True if GameState is a winning state
        gameState.isLose(): Return True if GameState is a losing state
        """

        def minimax(gameState: GameState, depth: int, agentIndex: int) -> float:
            # Base cases 
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Pac-Man's turn
            if agentIndex == 0:
                maxValue = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    value = minimax(successorState, depth - 1, 1)
                    maxValue = max(maxValue, value)
                return maxValue
            # Ghosts' turn
            else:
                # Last ghost's turn
                if agentIndex == gameState.getNumAgents() - 1:
                    minValue = float('inf')
                    for action in gameState.getLegalActions(agentIndex):
                        successorState = gameState.generateSuccessor(agentIndex, action)
                        value = minimax(successorState, depth - 1, 0)
                        minValue = min(minValue, value)
                    return minValue
                # Non-last ghost's turn
                else:
                    minValue = float('inf')
                    for action in gameState.getLegalActions(agentIndex):
                        successorState = gameState.generateSuccessor(agentIndex, action)
                        value = minimax(successorState, depth, agentIndex + 1)
                        minValue = min(minValue, value)
                    return minValue
        
        #  Choosing the best action
        bestScore = float('-inf')
        bestAction = None
    
        totalDepth = self.depth * (gameState.getNumAgents())
    
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            score = minimax(successorState, totalDepth - 1, 1)
        
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning
    """


    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action with alpha-beta pruning from the current gameState.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """
        def alphabeta(gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float) -> float:
            # Base cases 
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Pac-Man's turn 
            if agentIndex == 0:
                maxValue = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    value = alphabeta(successorState, depth - 1, 1, alpha, beta)
                    maxValue = max(maxValue, value)
                    alpha = max(alpha, maxValue)
                
                    # Pruning
                    if beta <= alpha:
                        break
                return maxValue

            # Ghosts' turn 
            else:
                # Last ghost's turn
                if agentIndex == gameState.getNumAgents() - 1:
                    minValue = float('inf')
                    for action in gameState.getLegalActions(agentIndex):
                        successorState = gameState.generateSuccessor(agentIndex, action)
                        value = alphabeta(successorState, depth - 1, 0, alpha, beta)
                        minValue = min(minValue, value)
                        beta = min(beta, minValue)
                    
                        # Pruning
                        if beta <= alpha:
                            break
                    return minValue
            
                # Non-last ghost's turn
                else:
                    minValue = float('inf')
                    for action in gameState.getLegalActions(agentIndex):
                        successorState = gameState.generateSuccessor(agentIndex, action)
                        value = alphabeta(successorState, depth, agentIndex + 1, alpha, beta)
                        minValue = min(minValue, value)
                        beta = min(beta, minValue)
                    
                        # Pruning
                        if beta <= alpha:
                         break
                    return minValue

        # Choosing the best action
        bestScore = float('-inf')
        bestAction = None
    
        totalDepth = self.depth * (gameState.getNumAgents())
    
        alpha = float('-inf')
        beta = float('inf')
    
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            score = alphabeta(successorState, totalDepth - 1, 1, alpha, beta)
        
            if score > bestScore:
                bestScore = score
                bestAction = action
        
            alpha = max(alpha, score)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent
    """

    def getAction(self, gameState):
        """Return the expectimax action from the current gameState.

        All ghosts should be modeled as choosing uniformly at random from their legal moves.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """
        def expectimax(gameState: GameState, depth: int, agentIndex: int) -> float:
            # Base cases
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Pac-Man's turn 
            if agentIndex == 0:
                maxValue = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    value = expectimax(successorState, depth - 1, 1)
                    maxValue = max(maxValue, value)
                return maxValue

            # Ghosts' turn
            else:
                # Last ghost's turn
                if agentIndex == gameState.getNumAgents() - 1:
                    legalActions = gameState.getLegalActions(agentIndex)
                    expectedValue = 0.0
                    for action in legalActions:
                        successorState = gameState.generateSuccessor(agentIndex, action)
                        value = expectimax(successorState, depth - 1, 0)
                        expectedValue += value / len(legalActions)
                    return expectedValue
            
                # Non-last ghost's turn
                else:
                    legalActions = gameState.getLegalActions(agentIndex)
                    expectedValue = 0.0
                    for action in legalActions:
                        successorState = gameState.generateSuccessor(agentIndex, action)
                        value = expectimax(successorState, depth, agentIndex + 1)
                        expectedValue += value / len(legalActions)
                    return expectedValue

        # Choosing best action
        bestScore = float('-inf')
        bestAction = None
    
        totalDepth = self.depth * (gameState.getNumAgents())
    
        for action in gameState.getLegalActions(self.index):
            successorState = gameState.generateSuccessor(self.index, action)
            score = expectimax(successorState, totalDepth - 1, 1)
        
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(gameState: GameState) -> float:
    """
    Return score of gameState using custom evaluation function that improves agent performance.
    """

    """
    The evaluation function takes the current GameStates (pacman.py) and returns a number,
    where higher numbers are better.

    Some methods/functions that may be useful for extracting game state:
    gameState.getPacmanPosition() # Pac-Man position
    gameState.getGhostPositions() # List of ghost positions
    gameState.getFood().asList() # List of positions of current food
    gameState.getCapsules() # List of positions of current capsules
    gameState.getGhostStates() # List of ghost states, including if current scared (via scaredTimer)
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    """

    # TODO: Implement your evaluation function
    raise Exception("Not implemented yet")


# Create short name for custom evaluation function
better = betterEvaluationFunction
