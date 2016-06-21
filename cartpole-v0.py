import time
import numpy as np
import gym

q_states = {}
alpha = 0.5
gamma = 0.25
best_ever = 0

def q_func(s, a, r, s_p):
    if s_p not in q_states.keys():
        q_states[s_p] = {}
        for act in xrange(env.action_space.n):
            q_states[s_p][act] = 1
    next_best = max(q_states[s_p].values())
    q_states[s][a] = (1.0 - alpha) * q_states[s][a] + alpha * (r + gamma * next_best)
    
def get_action(s, env):
    if s in q_states.keys():
        # default action is random if nothing is better.
        best = -1
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
        # 
        # take a random actions if you've never felt like this before.
        action = env.action_space.sample()
        q_states[s] = {}
        for act in xrange(env.action_space.n):
            q_states[s][act] = 1

    return action


        
good_results = 0
env = gym.make('CartPole-v0')
#env.monitor.start('/tmp/cartpole-experiment-3')
for i_episode in range(10000):
    
    observation = env.reset()
    state = tuple(observation > 0)
    score = 0
    #alpha = alpha * 0.95
    
    for t in range(500):
        # Save the previous state.
        prev_state = state
        
        #env.render()
        action = get_action(prev_state, env)#env.action_space.sample()
        observation, reward, done, info = env.step(action)

        state = tuple(observation > 0)
        if done:
            reward = score - 65
        q_func(prev_state, action, reward, state)
        #print(observation)

        score = score + 1

        #time.sleep(0.05)
        if done:
            if score > 75:
                #best_ever = score
                alpha = alpha * 0.85

            #best_ever *= 0.99
            if score >= 195:
                good_results = good_results + 1

            if (i_episode + 1) % 50 == 0:
                print '{} of {}'.format(good_results, i_episode + 1)

            #print("{}, a = {} finished after {} timesteps".format(i_episode, alpha, t+1))
            score = 0
            #print q_states
            break
        
    if i_episode == 10000:
        print q_states
        break
#env.monitor.close()
