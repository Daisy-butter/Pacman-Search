# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)  # 新状态
        newPos = successorGameState.getPacmanPosition()  # 新状态下pacman位置
        newFood = successorGameState.getFood()  # 新状态下食物矩阵
        newGhostStates = successorGameState.getGhostStates()  # 新状态下所有ghost状态
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  # 新状态下，吃掉ghost需要的移动数

        # return successorGameState.getScore()

        "*** YOUR CODE HERE ***"
        cuurent_pos = currentGameState.getPacmanPosition()
        # 首先进行胜利判断
        if successorGameState.isWin():
            return 99999

        from util import manhattanDistance
        # 新状态剩余食物的数量和距离
        newfood_list = newFood.asList()
        newfood_dist = []
        for food in newfood_list:
            newfood_dist.append(manhattanDistance(newPos, food))
        # newfood_dist.sort()

        newfood_left = len(newfood_list)

        # 当前状态剩余食物的数量和距离
        currentfood_list = currentGameState.getFood().asList()
        currentfood_dist = []
        for food in currentfood_list:
            currentfood_dist.append(manhattanDistance(cuurent_pos, food))
        currentfood_left = len(currentfood_list)

        # 新状态下所有鬼怪的位置和距离
        new_ghost_pos = []
        for ghost in newGhostStates:
            new_ghost_pos.append(ghost.getPosition())

        new_ghost_dist = []
        for pos in new_ghost_pos:
            new_ghost_dist.append(manhattanDistance(newPos, pos))

        # 当前状态下所有鬼怪的位置和距离
        current_ghost_pos = []
        for ghost in currentGameState.getGhostStates():
            current_ghost_pos.append(ghost.getPosition())

        current_ghost_dist = []
        # 当前状态下pacman位置
        for pos in current_ghost_pos:
            current_ghost_dist.append(manhattanDistance(cuurent_pos, pos))

        # 初始化分数
        score = 0
        # 原始分数
        score += successorGameState.getScore() - currentGameState.getScore()

        # 停顿惩罚
        if action == Directions.STOP:
            score -= 10

        # 吃掉大食物奖励
        if newPos in currentGameState.getCapsules():
            score += 150

        # 吃掉小食物奖励
        if currentfood_left > newfood_left:
            score += 200

        # 剩余食物的惩罚
        score -= 10 * newfood_left

        def reward_dist(dist):
            if dist == 0:
                return 200
            else:
                return int(200 / dist)

        # 距离食物越近越好
        score += reward_dist(min(newfood_dist)) - reward_dist(min(currentfood_dist))

        def penalty_dist(dist):
            if dist == 0:
                return 99999
            else:
                return int(1000 / dist)

        sum_ScaredTimes = sum(newScaredTimes)
        if sum_ScaredTimes > 0:
            # 距离鬼怪越近越好
            score -= penalty_dist(min(current_ghost_dist)) - penalty_dist(min(new_ghost_dist))
        else:

            score += penalty_dist(min(current_ghost_dist)) - penalty_dist(min(new_ghost_dist))

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def terminalTest(gameState, depth):
            return depth == self.depth or gameState.isWin() or gameState.isLose()

        def maximizer(currentState, depth):
            # 如果当前状态是终止状态或达到指定深度，则评估当前状态并返回分数
            if terminalTest(currentState, depth):
                return self.evaluationFunction(currentState)
            # 初始化最大分数为负无穷，确保任何合法动作的分数都会大于初始值
            maxScore = float("-inf")
            "*** YOUR CODE HERE ***"
            for action in currentState.getLegalActions(0):
                successor = currentState.generateSuccessor(0, action)
                maxScore = max(maxScore, minimizer(successor, depth, 1))
            return maxScore

        def minimizer(currentState, depth, ghostIndex):
            # 检查是否达到终止条件，如果是则返回当前状态的评估值
            if terminalTest(currentState, depth):
                return self.evaluationFunction(currentState)
            # 初始化最小得分为正无穷，确保任何有效得分都会比它小
            minScore = float("inf")
            "*** YOUR CODE HERE ***"
            # 遍历当前ghost的所有合法行动
            for action in currentState.getLegalActions(ghostIndex):
                # 生成执行该行动后的后继状态
                successor = currentState.generateSuccessor(ghostIndex, action)
                # 如果是最后一个ghost，则增加深度并调用maximizer（玩家回合）
                if ghostIndex == numAgents - 1:
                    minScore = min(minScore, maximizer(successor, depth + 1))
                # 否则，调用下一个ghost的minimizer
                else:
                    minScore = min(minScore, minimizer(successor, depth, ghostIndex + 1))
            # 返回所有可能行动中的最小得分
            return minScore

        numAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(0)
        bestAction = legalActions[0]
        "*** YOUR CODE HERE ***"
        bestScore = float("-inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = minimizer(successor, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def terminalTest(gameState, depth):
            return depth == self.depth or gameState.isWin() or gameState.isLose()

        def maximizer(currentState, depth, alpha, beta):
            if terminalTest(currentState, depth):
                return self.evaluationFunction(currentState)
            maxScore = float("-inf")
            alpha1 = alpha
            for action in currentState.getLegalActions(0):
                successor = currentState.generateSuccessor(0, action)
                maxScore = max(maxScore, minimizer(successor, depth, 1, alpha1, beta))
                if maxScore > beta:
                    return maxScore
                alpha1 = max(alpha1, maxScore)
            return maxScore

        def minimizer(currentState, depth, ghostIndex, alpha, beta):
            if terminalTest(currentState, depth):
                return self.evaluationFunction(currentState)
            minScore = float("inf")
            beta1 = beta
            for action in currentState.getLegalActions(ghostIndex):
                successor = currentState.generateSuccessor(ghostIndex, action)
                if ghostIndex == numAgents - 1:
                    minScore = min(minScore, maximizer(successor, depth + 1, alpha, beta1))
                    if minScore < alpha:
                        return minScore
                    beta1 = min(beta1, minScore)

                else:
                    minScore = min(minScore, minimizer(successor, depth, ghostIndex + 1, alpha, beta1))
                    if minScore < alpha:
                        return minScore
                    beta1 = min(beta1, minScore)

            return minScore

        legalActions = gameState.getLegalActions(0)
        bestAction = legalActions[0]
        bestScore = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = minimizer(successor, 0, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action

            if score > beta:
                return bestAction
            alpha = max(alpha, score)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # 获取游戏中的智能体数量
        numAgents = gameState.getNumAgents()

        # 终止测试函数，检查是否达到指定深度或游戏结束
        def terminalTest(gameState, depth):
            return depth == self.depth or gameState.isWin() or gameState.isLose()

        # 最大化玩家（智能体0）的得分
        def maximizer(currentState, depth):
            # 如果达到终止条件，返回评估函数值
            if terminalTest(currentState, depth):
                return self.evaluationFunction(currentState)
            # 初始化最大得分为负无穷
            maxScore = float("-inf")
            # 遍历所有合法动作
            for action in currentState.getLegalActions(0):
                # 生成后继状态
                successor = currentState.generateSuccessor(0, action)
                # 递归计算期望值，并更新最大得分
                maxScore = max(maxScore, expectimin(successor, depth, 1))
            return maxScore

        # 计算期望值函数，处理幽灵的移动
        def expectimin(currentState, depth, ghostIndex):
            # 如果达到终止条件，返回评估函数值
            if terminalTest(currentState, depth):
                return self.evaluationFunction(currentState)
            # 初始化期望得分为0
            expScore = 0
            # 获取当前幽灵的合法动作列表
            action_list = currentState.getLegalActions(ghostIndex)
            # 遍历所有可能的动作
            for action in action_list:
                # 生成后继状态
                successor = currentState.generateSuccessor(ghostIndex, action)
                # 如果是最后一个幽灵，则递归调用最大化函数
                if ghostIndex == numAgents - 1:
                    expScore += (1 / len(action_list)) * maximizer(successor, depth + 1)
                # 否则，递归调用期望值函数处理下一个幽灵
                else:
                    expScore += (1 / len(action_list)) * expectimin(successor, depth, ghostIndex + 1)
            return expScore

        # 获取玩家的合法动作列表
        legalActions = gameState.getLegalActions(0)
        # 初始化最佳动作为第一个合法动作
        bestAction = legalActions[0]
        # 初始化最佳得分为负无穷
        bestScore = float("-inf")
        # 遍历所有合法动作
        for action in legalActions:
            # 生成后继状态
            successor = gameState.generateSuccessor(0, action)
            # 计算该动作的得分
            score = expectimin(successor, 0, 1)
            # 如果当前得分更高，则更新最佳动作和得分
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    from util import manhattanDistance  # 导入曼哈顿距离计算工具

    # 如果游戏胜利，返回一个很高的分数
    if currentGameState.isWin():
        return 999999
    # 如果游戏失败，返回一个很低的分数
    if currentGameState.isLose():
        return -999999

    # 获取吃豆人当前位置
    pacman_pos = currentGameState.getPacmanPosition()
    # 获取食物网格
    food_grid = currentGameState.getFood()
    # 将食物网格转换为食物列表
    food_list = food_grid.asList()
    # 计算吃豆人到所有食物的距离
    food_dist = []
    for food in food_list:
        food_dist.append(manhattanDistance(pacman_pos, food))
    # 剩余食物数量
    food_left = len(food_list)

    # 获取所有鬼怪的恐惧时间
    scared_times = [ghost.scaredTimer for ghost in currentGameState.getGhostStates()]

    # 分离被吓到的鬼怪和未被吓到的鬼怪的位置
    scared_ghost_pos = []
    unscared_ghost_pos = []
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer > 0:
            scared_ghost_pos.append(ghost.getPosition())
        else:
            unscared_ghost_pos.append(ghost.getPosition())

    # 计算吃豆人到被吓到的鬼怪和未被吓到的鬼怪的距离
    scared_ghost_dist = []
    unscared_ghost_dist = []
    for pos in scared_ghost_pos:
        scared_ghost_dist.append(manhattanDistance(pacman_pos, pos))
    for pos in unscared_ghost_pos:
        unscared_ghost_dist.append(manhattanDistance(pacman_pos, pos))

    # 被吓到的鬼怪和未被吓到的鬼怪的数量
    scared_ghost_left = len(scared_ghost_pos)
    unscared_ghost_left = len(unscared_ghost_pos)

    # 获取能量豆的位置
    capsule_pos = currentGameState.getCapsules()
    # 计算吃豆人到所有能量豆的距离
    capsule_dist = []
    for pos in capsule_pos:
        capsule_dist.append(manhattanDistance(pacman_pos, pos))

    # 剩余能量豆数量
    left_capsule = len(capsule_pos)

    # 定义距离奖励函数
    def reward_dist(dist):
        if dist < 1:
            return 1000
        else:
            return 1 / dist

    # 定义距离惩罚函数
    def penalty_dist(dist):
        if dist == 0:
            return 999999
        elif dist <= 1:
            return 1000
        else:
            return 0

    # 获取当前游戏分数
    score = currentGameState.getScore()

    # 靠近食物的奖励
    if left_capsule > 0:
        score += reward_dist(min(capsule_dist))  # 靠近能量豆的奖励
    if food_left > 0:
        aver_dist = sum(food_dist) / len(food_dist)  # 到所有食物的平均距离
        score += reward_dist(aver_dist) + reward_dist(min(food_dist))  # 平均距离和最近距离的奖励

    if scared_ghost_left > 0:
        # 靠近被吓到的鬼怪的奖励
        score += sum([reward_dist(dist) for dist in scared_ghost_dist])

    if unscared_ghost_left > 0:
        # 远离未被吓到的鬼怪的惩罚
        score -= penalty_dist(min(unscared_ghost_dist))
    return score


better = betterEvaluationFunction
