###### These notes are for whoever would like to understand Q-Learning and especially for me, to make sure not forgeting any notion here.
# CartPole-Q-Learning

## Running the code

Only two library are needed to make the code compile : ```gym``` and ```numpy```
On a terminal, run the following commands :

```
$ pip install gym==0.25.2
$ pip install numpy
```

So the only imports to run the code will be these two libraries. Now just run the code in your IDE.
In the imports section you will find the ```matplotlib``` library, this one is unecessary, it's useful if you want to track the evolution of the score depending on the episodes. I wrote some comments next to lines that are using the library, don't hesitate to remove these lines.
## How it works

#### Q-Learning
In *Q-Learning*, we have a whole environment with differents actors and variables, let's explain them. To summarize *Q-Learning*, let us introduce an agent, that makes decisions over a environment $E$, these desicions are called actions $A$ and are made depending a state $S$. Each of these actions produce events, consequences. The agent (here the cart) will evaluate whether the last action added value on the environment or not. Depending on the answer, the agent will reward the environment and will base the next computation on this reward.

More precisely, *Q-Learning* learns a $Q$ function. This function estimate the potential gain, it means the sum of rewards $Q(S, A)$ on long term. So before the learning phase begins, the $Q$ function is initialized randomly, then, with each choice of action, the agent observes the reward and the new state and compute the new $Q$ : 

$$Q_{\text{new}}(S, A) = (1 - \alpha)Q(S,A) + \alpha(r + \gamma(maxQ(s', a')))$$
with $r$ the reward, $s'$ the new state, $\alpha$ the learning rate and $\gamma$ the discount factor that represent at what point we care or not about the reward in the calculus.

#### The code
This code use Q-Learning to workout the perfect balanced pole.

First of all, we initialize the Q-Table. The issue here is that the cart and the pole only give su continous values, as the pole and the cart can be in different states depending on cart position, angle, etc...
So we want to use exploitable values, that's why we compute discretes values, it will make the calcultations easier. Bins are ranges of values that our four states can take :
- cart position
- cart velocity
- pole angle
- pole velocity

Then we have the main function ```fit```. Let us break it down. Basically, we train our model a certain amount of time, so it can learns from the rewards. For each episode (iteration), we're balancing the pole while the condition ```done``` is false, this condition check for each action we're doing if the pole is more than 15 degrees from vertical or the cart moves more than 2.4 units from the center. Depending on the ouput of the action, the score get rewarded or not. Then, if the condition is still false, we update the $Q$, based on the output of our last action.



