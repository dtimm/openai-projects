import time
import random
import numpy as np
import gym

q_states = {}
alpha = 0.5
gamma = 0.25
random_act = 0.5

def q_func(s, a, r, s_p):
    if s_p not in q_states.keys():
        q_states[s_p] = {}
        for act in xrange(env.action_space.n):
            q_states[s_p][act] = 1

    next_best = max(q_states[s_p].values())
    q_states[s][a] = (1.0 - alpha) * q_states[s][a] + alpha * (r + gamma * next_best)
    
def get_action(s, env):

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
        # 
        # take a random actions if you've never felt like this before.
        action = env.action_space.sample()
        q_states[s] = {}
        for act in xrange(env.action_space.n):
            q_states[s][act] = 1

    return action


avg_score = 0
good_results = 0
env = gym.make('CartPole-v0')
#env.monitor.start('tmp/cartpole-experiment-1')
for i_episode in range(10001):
    
    observation = env.reset()
    state = ()
    for val in observation:
        state += (round(val * 2.5)/2.5, )
    score = 0
    #alpha = alpha * 0.95
    
    for t in range(500):
        # Save the previous state.
        prev_state = state
        
        #env.render()
        action = get_action(prev_state, env)#env.action_space.sample()
        observation, reward, done, info = env.step(action)

        state = ()
        for val in observation:
            state += (round(val * 2.5)/2.5, )
        
        if reward > 1.0:
            print reward
            
        if done:
            reward = score - 200
        q_func(prev_state, action, reward, state)
        #print(observation)

        score = score + 1

        #time.sleep(0.05)
        if done:
            avg_score += score

            #if score > 75:
                #best_ever = score
                #alpha = alpha * 0.85

            if score >= 195:
                good_results = good_results + 1

            if (i_episode + 1) % 50 == 0:
                print '{0:.00%} at {1}'.format(float(good_results) / 50.0, i_episode + 1)
                print '{0} average score'.format(float(avg_score) / 50.0)
                
                random_act *= 0.9
                avg_score = 0
                good_results = 0

            #print("{}, a = {} finished after {} timesteps".format(i_episode, alpha, t+1))
            score = 0
            #print q_states
            break
        
#env.monitor.close()