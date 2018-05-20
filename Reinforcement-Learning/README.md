# Learning with RL


## Udacity : Dynamic Programming

+ Getting to the optimal policy
  + One Step dynamics - Moving by one step
  + Agent calculates the value function through iterative policy Evaluation
  + Design the Policy Improvement process such that the new policy is at least as good as the current one
  + policy (Iterative policy evaluation) -> Value Function (policy Improvement) -> new Policy [Iterate till be converge]
+ TODO: not just the FrozenLake env. Try the optimal policy iteration for any MDP

+ **parts 0/1**

```

#actual solution
def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V

#mine
def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)

    ## TODO: complete the function
    while True:
            delta=0
            for i in range(env.nS): # For all states
                res_=0
                for a, prob in enumerate(policy[i]): # For all actions
                    for probv, next_state, reward, done in env.P[i][a]:
                        res_+= prob*probv*(reward+V[next_state]*gamma)
                    #res_=policy[i][j]*sum_
                delta=max(delta, np.abs(res_-V[i]))
                V[i]=res_
            if delta<theta:
                break
        return V

```

+ **parts 2**

```
# env.P[1][0] = prob, next_state, reward, done

def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)

    ## TODO: complete the function
    for a in range(env.nA):
        for x in range(len(env.P[s][a])):
            q[a]+= env.P[s][a][x][0]*(env.P[s][a][x][2] + gamma*V[env.P[s][a][x][1]])
    return q
```

### TD Learning and Q-Learning
+ TD(0)
+ Sarsa-0/Q-learning(0)
+ Sarsamax or Q-learning
+ Expected Sarsamax

### **Max-q Method** [Hierarchical RL with MAX-Q Function -Decomposition] [State abstraction to achieve better Performance] [The Taxi Domain RL Problem]

+ Reference: https://arxiv.org/pdf/cs/9905014.pdf
+ Max-q method , provides a Hierarchical decomposition of the given RL problem into sub-problems
  - Given value Function into a number of sub-value functions
+ Three ways to break the Target MDP into hierarchy of Sub-problems
  - One approach is to define each subtask as a fixed policy that is provided to the programmer **[ The “option” method of Sutton,Precup, and Singh (1998) takes this approach]**
  - Define each subtask in terms of a non-deterministic Finite state controller **[ Hierarchy of Abstract Machines (HAM) method of Parr and Russell (1998)]**
    + This allows the programmer to provide a "Partial Policy" that constrains the set of the permitted actions at each point, but does not specify any complete policy
  - Define each subtask in terms of termination predicate and a local reward function.  Final reward  
+ Problems with this Hierarchical Approach (3rd approach)
  - The termination predicate method suffers from an additional source of sub optimality. The learning algorithm described in this paper converges to a form of local optimality that we call
***recursive optimality***
+ We will see that the MAXQ method creates many opportunities to exploit state abstraction, and that these abstractions can have a huge impact in accelerating learning
+ The successful use of state abstraction requires that subtasks be defined in terms of termination predicates rather than using the option or partial policy methods
+ In this paper, The MAXQ-Q, does a fully online-learning of hierarchical value function, we show that the algorithm converges to a recursively optimal policy and is faster

## Resources
+ Richard Sutton's book : http://www.wildml.com/2016/10/learning-reinforcement-learning/
+ Medium - https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287
+ TD learning simulation - http://kldavenport.com/suttons-temporal-difference-learning/
+ Deep RL Learning - https://mpatacchiola.github.io/ | https://github.com/mpatacchiola/dissecting-reinforcement-learning
+ RL problem - http://www-anw.cs.umass.edu/~barto/courses/cs687/
