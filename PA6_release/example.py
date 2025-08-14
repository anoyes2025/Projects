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
       # TODO: Implement epsilon action selection
       if random.random() < self.epsilon:
           return random.randint(0, self.numActions - 1)
       else:
           return self.action(state)
       

   def learningStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
       """Perform a value-iteration learning step for the given transition and reward."""
   
       # Update the observed transitions and rewards for (s, a, s') triples. Since we are using
       # defaultdicts we don't need to check if the key exists before incrementing.
       self.tCounts[curState][action][nextState] += 1
       self.rTotal[curState][action][nextState] += reward

       # Update the current policy every updateIter steps
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
       v = [0.0] * self.numStates
       while True:
           v_new = v[:]

           # TODO: Calculate v_new for each state for which you have observed transitions
           for s in range (self.numStates):
               max_val = float('-inf')
               best_action = None
               for a in range (self.numActions):
                   total_transitions = sum(self.tCounts[s][a].values())
                   if total_transitions > 0:
                       val = sum((self.tCounts[s][a][s_] / total_transitions) * (self.rTotal[s][a][s_] / total_transitions) for s_ in self.tCounts[s][a]) + self.gamma * max(v[s_] for s_ in self.tCounts[s][a])
                       if val > max_val:
                           max_val = val
                           best_action = a
               v_new[s] = max_val
               self.pi[s] = best_action if best_action is not None else random.randint(0, self.numActions - 1)

           # Change in values?       
           if all(abs(new - prev) <= self.valueConvergence for new, prev in zip(v_new, v)):
               break
           v = v_new             
      
       # TODO: Update policy based on results of value iteration      

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
       # TODO: Implement greedy action selection
       max_q = max(self.q[state])
       # Get all actions that tie for maximum Q-value
       best_actions = [a for a, q in enumerate(self.q[state]) if q == max_q]
       # Randomly break ties
       return random.choice(best_actions)

   def epsilonAction(self, step: int, state: int) -> int:
       """With probability epsilon returns a uniform random action. Otherwise it returns a greedy action with respect to the current Q function (breaking ties randomly)."""
       # TODO: Implement epsilon-greedy action selection
       if random.random() < self.epsilon:
           return random.randint(0, self.numActions - 1)
       else:
           return self.action(state)

   def learningStep(self, step: int, curState, action, reward, nextState):
       """Performs a Q-learning step based on the given transition, action and reward."""
       # TODO: Implement the Q-learning step
       self.q[curState][action] += self.alpha * (reward + self.gamma * max(self.q[nextState]) - self.q[curState][action])


   def terminalStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
       """Performs the last learning step of an episode. Because the episode has terminated, the next Q-value is 0."""
       # TODO: Implement the terminal step of the learning algorithm
       self.q[curState][action] += self.alpha * (reward - self.q[curState][action])