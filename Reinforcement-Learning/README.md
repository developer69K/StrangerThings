# Learning with RL


## Udacity : Dynamic Programming

+ Getting to the optimal policy
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
def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)

    ## TODO: complete the function
    for a in range(env.nA):
        sum_=0
        for x in range(len(env.P[s][a])):
            sum_+= env.P[s][a][x][0]*(env.P[s][a][x][2] + gamma*V[env.P[s][a][x][1]])
        q[a]=sum_    
    return q
```

## Resources
+ Richard Sutton's book : http://www.wildml.com/2016/10/learning-reinforcement-learning/
