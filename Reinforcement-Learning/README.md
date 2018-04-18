# Learning with RL


## Dynamic Programming

+ **parts 0/1**
```(python)

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


## Resources
+ Richard Sutton's book : http://www.wildml.com/2016/10/learning-reinforcement-learning/
