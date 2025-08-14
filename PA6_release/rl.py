import abc
from collections import defaultdict
import random
from typing import Dict, List

class ReinforcementLearner(metaclass=abc.ABCMeta):
    """Represents an abstract reinforcement learning agent."""

    def __init__(self, numStates: int, numActions: int, epsilon: float, gamma: float, **kwargs):
        """Initialize GridWorld reinforcement learning agent.

        Args:
            numStates (int): Number of states in the MDP.
            numActions (int): Number of actions for each state in the MDP.
            epsilon (float): Probability of taking a random action.
            gamma (float): Discount parameter.
        """
        self.numStates = numStates
        self.numActions = numActions

        self.epsilon = epsilon
        self.gamma = gamma

        
    @abc.abstractmethod
    def action(self, state: int) -> int:
        """Return learned action for the given state."""
        pass

    @abc.abstractmethod
    def epsilonAction(self, step: int, state: int) -> int:
        """With probability epsilon returns a uniform random action. Otherwise return learned action for given state."""
        pass

    def terminalStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Perform the last learning step of an episode. 

        Args:
            step (int): Index of the current step.
            curState (int): Current state, e.g., s
            action (int): Current action, e.g., a
            reward (float): Observed reward
            nextState (int): Next state, e.g., s'. Since this is a terminal step, this is a terminal state.
        """
        self.learningStep(step, curState, action, reward, nextState)

    @abc.abstractmethod
    def learningStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Perform a learning step of an episode. 

        Args:
            step (int): Index of the current step.
            curState (int): Current state, e.g., s
            action (int): Current action, e.g., a
            reward (float): Observed reward
            nextState (int): Next state, e.g., s'.
        """
        pass


class ModelBasedLearner(ReinforcementLearner):
    """Model-based value iteration reinforcement learning agent."""
    def __init__(self, numStates: int, numActions: int, epsilon: float, gamma: float, updateIter: int = 1000, valueConvergence: float = .001, **kwargs):
        super().__init__(numStates, numActions, epsilon, gamma)
        
        self.updateIter = updateIter
        self.valueConvergence = valueConvergence
        self.change = float('inf')

        # Maintain transition counts and total rewards for each (s, a, s') triple as a list-of-lists-of dictionaries
        # indexed first by state, then by actions. The keys are in the dictionaries are s'.
        self.tCounts: List[List[defaultdict]] = []
        self.rTotal : List[List[defaultdict]]= []
        for _ in range(numStates):
            self.tCounts.append([defaultdict(int) for _ in range(numActions)])
            self.rTotal.append([defaultdict(float) for _ in range(numActions)])

        # Current policy implemented as a dictionary mapping states to actions. Only states with a current policy
        # are in the dictionary. Other states are assumed to have a random policy.
        self.pi: Dict[int, int] = {}

    def action(self, state: int) -> int:
        """Return the action in the current policy for the given state."""
        # Return the specified action in the current policy if it exists, otherwise return
        # a random action
        return self.pi.get(state, random.randint(0, self.numActions - 1))
    

    def epsilonAction(self, step: int, state: int) -> int:
        """With some probability return a uniform random action. Otherwise return the action in the current policy for the given state."""
        # Boost epsilon for steps <3000, or decay
        if step <= 3000:
            if random.random() < 3: # 5 as a boosted epsilon value
                return random.randint(0, self.numActions - 1)
            return self.action(state)
        else:
            decayed_epsilon = self.epsilon * (0.99 ** (step - 3000))
            if random.random() < decayed_epsilon:
                return random.randint(0, self.numActions - 1)
            return self.action(state)

    def learningStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Perform a value-iteration learning step for the given transition and reward."""

        # Update transition counts and total rewards
        self.tCounts[curState][action][nextState] += 1
        self.rTotal[curState][action][nextState] += reward

        # Update policy every updateIter steps
        if step % self.updateIter != 0:
            return
        
        # Implement value iteration to update the policy. 
        # Recall that:
        #   T(s, a, s') = (Counts of the transition (s,a) -> s') / (total transitions from (s,a))
        #   R(s, a, s') = (total reward of (s,a) -> s') / (counts of transition (s,a) -> s')
        # Many states may not have been visited yet, so we need to check if the counts are zero before
        # updating the policy. We will only update the policy for states with state-action pairs that\
        # have been visited.
  
        # Recall value iteration is an iterative algorithm. Here iterate until convergence, i.e., when
        # the change between v_new and v is less than self.valueConvergence for all states.

        # Initialize value function
        v = [0.0] * self.numStates

        # Iterate until convergence
        while True:
            v_save = v[:]
            for state in range(self.numStates):
                max_val = -float('inf')
                best_action = None

                for action in range(self.numActions):
                    total_transitions = sum(self.tCounts[state][action].values())
                    if total_transitions != 0:
                        new_val = sum((self.tCounts[state][action][sp] / total_transitions) * (self.rTotal[state][action][sp] / total_transitions + self.gamma * v[sp]) for sp in self.tCounts[state][action])
                        if new_val > max_val:
                            max_val = new_val
                            best_action = action

                # Only update if transitions were observed
                if max_val != float('-inf'):
                    v_save[state] = max_val
                    self.pi[state] = best_action

            if all(abs(new - old) <= self.valueConvergence for new, old in zip(v_save, v)):
               break
            v = v_save
            
        
class QLearner(ReinforcementLearner):
    """Q-learning-based reinforcement learning agent."""
    
    def __init__(self, numStates: int, numActions: int, epsilon: float, gamma: float, alpha: float = 0.1, initQ: float=0.0, **kwargs):
        """Initialize GridWorld reinforcement learning agent.

        Args:
            numStates (int): Number of states in the MDP.
            numActions (int): Number of actions for each state in the MDP.
            epsilon (float): Probability of taking a random action.
            gamma (float): Discount parameter.
            alpha (float, optional): Learning rate. Defaults to 0.1.
            initQ (float, optional): Initial Q value. Defaults to 0.0.
        """
        super().__init__(numStates, numActions, epsilon=epsilon, gamma=gamma)

        self.alpha = alpha

        # The Q-table, q, is a list-of-lists, indexed first by state, then by actions
        self.q: List[List[float]] = []  
        for _ in range(numStates):
            self.q.append([initQ] * numActions)

    def action(self, state: int) -> int:
        """Returns a greedy action with respect to the current Q function (breaking ties randomly)."""
        q_values = self.q[state]
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def epsilonAction(self, step: int, state: int) -> int:
        """With probability epsilon returns a uniform random action. Otherwise it returns a greedy action with respect to the current Q function (breaking ties randomly)."""
        # Use a single random number to decide between epsilon-greedy and greedy actions
        if random.random() < self.epsilon:
            return random.randint(0, self.numActions - 1)
        return self.action(state)

    def learningStep(self, step: int, curState, action, reward, nextState):
        """Performs a Q-learning step based on the given transition, action and reward."""
        next_max_q = max(self.q[nextState])
        current_q = self.q[curState][action]
        self.q[curState][action] += self.alpha * (reward + self.gamma * next_max_q - current_q)

    def terminalStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Performs the last learning step of an episode. Because the episode has terminated, the next Q-value is 0."""
        self.q[curState][action] = (1 - self.alpha) * self.q[curState][action] + self.alpha * reward
