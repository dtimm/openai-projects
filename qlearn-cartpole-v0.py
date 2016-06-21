import time
import random
import numpy as np
import gym

# Set initial parameter values.
q_states = {}
alpha = 0.4
gamma = 0.25
random_act = 0.5

# Run the Q function to update states.
def q_func(s, a, r, s_p):
    if s_p not in q_states.keys():
        q_states[s_p] = {}
        for act in xrange(env.action_space.n):
            q_states[s_p][act] = 1

    next_best = max(q_states[s_p].values())
    q_states[s][a] = (1.0 - alpha) * q_states[s][a] + alpha * (r + gamma * next_best)
    
# Get the next action.
def get_action(s, env):
    # Occasionally randomness to break it out of a funk.
    if random.random() < random_act:
        return env.action_space.sample()

    if s in q_states.keys():
        # default action is random if nothing is better.
        best = -10
        best_act = env.action_space.sample()
        for k, v in q_states[s].iteritems():
            if v > best:
                best = v
                best_act = k

        if best_act == 'none':
            action = None
        else:
            action = best_act
    else:
        # Take a random actions if you've never felt like this before.
        action = env.action_space.sample()

        # Initialize this new state.
        q_states[s] = {}
        for act in xrange(env.action_space.n):
            q_states[s][act] = 1

    return action

# Initial values and logging.
avg_score = 0
good_results = 0
env = gym.make('CartPole-v0')
filename = 'tmp/cartpole-experiment'
env.monitor.start(filename, force=True)

for i_episode in range(10001):
    # Reset and note initial state.
    observation = env.reset()

    # State is represented with binned states independent of the environment.
    state = ()
    for val in observation:
        state += (round(val * 4.0)/4.0, )
    
    # Run 200 time steps
    for t in range(200):
        # Save the previous state.
        prev_state = state
        
        # Pick an action and step forward.
        #env.render()
        action = get_action(prev_state, env)
        observation, reward, done, info = env.step(action)

        # Format the new state.
        state = ()
        for val in observation:
            state += (round(val * 4.0)/4.0, )

        # Reward agents that meet the threshhold for success, punish others.      
        if done:
            reward = t - 195

        # Update Q states.
        q_func(prev_state, action, reward, state)
        
        # Check it it's done enough.
        if done or t == 199:
            # Track average scores each 100 trials.
            avg_score += t

            # Log successes
            if t >= 195:
                good_results = good_results + 1
                # Every success reduced the chance of random actions.
                random_act *= 0.95
            
            # Print some debugging/logging information.
            if (i_episode + 1) % 100 == 0:
                print '{0:.00%} at {1}'.format(float(good_results) / 100.0, i_episode + 1)
                print '{0} average score'.format(float(avg_score) / 100.0)
                
                # Reduce randomness every 100 trials.
                random_act *= 0.95

                # Reset per 100 variables.
                avg_score = 0
                good_results = 0

            break
        
env.monitor.close()

