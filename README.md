# POLICY ITERATION ALGORITHM

## AIM
The aim of the experiment is to explore and deepen our understanding of the policy evaluation and policy iteration functions for a given environment and study about their impact on the value function.

## PROBLEM STATEMENT
The Problem statement involves taking a Frozen Lake Markov Decision Process and applying the policy evaluation and policy iteration algorithm on them and study its effect on the state-value function of the policy function

## POLICY ITERATION ALGORITHM
Include the steps involved in policy iteration algorithm
The algorithm implemented in the policy_iteration is a method used to find the optimal policy in a Markov decision process (MDP).

Step 1 : Initialize the policy pi. In this implementation, a random action is chosen for each state s in the MDP P. The initial policy is represented by the lambda function pi=lambda s:{s:a for s,a in enumerate(random_actions)}[s], where random_actions is a list of randomly chosen actions for each state.

Step 2 : Enter a loop that continues until the policy pi is no longer changing. This is determined by comparing the previous policy (old_pi) with the current policy computed in the loop.

Step 3 : Store the previous policy as old_pi for comparison later.

Step 4 : Perform policy evaluation using the function policy_evaluation. This step calculates the state-values (V) for each state s given the current policy pi. The state-values represent the expected cumulative rewards starting from state s following policy pi and discounting future rewards by a factor of gamma. The function policy_evaluation is called with the arguments pi, P, gamma, and theta.

Step 5 : Perform policy improvement using the function policy_improvement. This step updates the policy pi based on the current state-values V. The function policy_improvement is called with the arguments V, P, and gamma.

Step 6 : Check if the policy has converged by comparing the previous policy old_pi with the current policy {s:pi(s) for s in range(len(P))}. If they are the same for all states s, the loop is exited.

Step 7 : Return the final state-values V and the optimal policy pi.

To summarize, policy iteration iteratively improves the policy by alternating between policy evaluation and policy improvement steps until convergence is reached. The algorithm guarantees to find the optimal policy for the given MDP P with a discount factor gamma.

## POLICY IMPROVEMENT FUNCTION
### Name: Priyadharshan S
### Register Number: 212223240127
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:

          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))

        new_pi = lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
### Name: Priyadharshan S
### Register Number: 212223240127
```python
def policy_iteration(P, gamma=1.0,theta=1e-10):
  random_actions=np.random.choice(tuple(P[0].keys()),len(P))
  pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
  while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
  return V,pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="555" height="168" alt="image" src="https://github.com/user-attachments/assets/98d95c96-850e-4d53-91fa-9ec34e555798" />
<img width="674" height="39" alt="image" src="https://github.com/user-attachments/assets/8cff1549-257f-4152-8943-06eb67eb6e8f" />

### 2. Policy, Value function and success rate for the Improved Policy
<img width="527" height="176" alt="image" src="https://github.com/user-attachments/assets/620aca5b-3447-4b5e-898b-f9bcd860f19d" />
<img width="721" height="328" alt="image" src="https://github.com/user-attachments/assets/8184a140-16d5-497c-9fb9-6a5fb39ebad1" />

### 3. Policy, Value function and success rate after policy iteration
<img width="584" height="427" alt="image" src="https://github.com/user-attachments/assets/ba1dfc52-49e8-4215-8a24-b3c2727e90aa" />
<img width="880" height="435" alt="image" src="https://github.com/user-attachments/assets/024abc7f-51cc-4fb9-8450-bf8efadaeb87" />

## RESULT:

Thus we have successfully implemented policy iteration and policy improvement.
