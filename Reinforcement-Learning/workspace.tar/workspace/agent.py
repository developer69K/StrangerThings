import numpy as np
from collections import defaultdict

'''
root@1b729fe04e4d:/home/workspace# python main.py
Episode 20000/20000 || Best average reward 9.658

root@1b729fe04e4d:/home/workspace# python main.py
Episode 20000/20000 || Best average reward 9.314

root@1b729fe04e4d:/home/workspace# python main.py
Episode 20000/20000 || Best average reward 9.325

Episode 20000/20000 || Best average reward 9.166

root@1b729fe04e4d:/home/workspace# python main.py
Episode 20000/20000 || Best average reward 9.246

root@1b729fe04e4d:/home/workspace# python main.py
Episode 20000/20000 || Best average reward 9.361

root@1b729fe04e4d:/home/workspace# python main.py
Episode 19970/20000 || Best average reward 9.792
Environment solved in 19970 episodes.root@1b729fe04e4d:/home/workspace#
'''

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha=0.85
        self.gamma=0.98
        self.epsilon=0.005
        self.i_episode=1
        self.policy_s=None
    
    def generate_episode_from_Q(self, env):
        episode=[]
        state=env.reset()
        while True:
            action=self.select_action(state)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state=next_state
            if done:
                break
        return episode        
    
    # greedu-Eplsilon on the Limit of exploration
    def get_probs(self, Q_s, i_episode, eps=None):
        #epsilon=1.0/np.sqrt((9500*i_episode))
        epsilon=1.08/np.sqrt((9500*i_episode))
        if eps is not None:
            epsilon=eps
        policy_s=np.ones(self.nA) * epsilon / self.nA
        best_a = np.argmax(Q_s)
        policy_s[best_a]=1-epsilon+(epsilon/self.nA)
        return policy_s
    
    def Update_Q(self, Q_sa, Q_sa_next, reward):
        return Q_sa + (self.alpha *(reward + (self.gamma*Q_sa_next) - Q_sa))
    
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.policy_s=self.get_probs(self.Q[state],self.i_episode,eps=None)
        self.i_episode+=1
        action = np.random.choice(np.arange(self.nA),p=self.policy_s)
        return action
        #return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] = self.Update_Q(self.Q[state][action],np.dot(self.Q[next_state], self.policy_s), reward) 
        #self.Q[state][action] += 1
        