import random
import gym
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def state_from_observation(observation):
    state = ()
    for val in observation:
        temp_value = round(val * 4.0)/4.0
        if temp_value == -0.0:
            temp_value = 0.0
        state += (temp_value, )
    return state

def main(argv=None):
    # Initial values and logging.
    q_states = {}
    alpha = 0.5
    gamma = 0.95
    avg_score = 0
    good_results = 0
    goal_score = 195
    env = gym.make('CartPole-v0')
    filename = 'tmp/cartpole-experiment'
    env.monitor.start(filename, force=True)

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
    
        if s in q_states.keys():
            # default action is random if nothing is better.
            values = softmax(q_states[s].values())
            if random.random() < values[0]:
                action = 0
            else:
                action = 1
            #action = max(q_states[s], key=q_states[s].get)
        else:
            # Find the most similar state
            best_diff = -1.0
            best_state = None
            for state in q_states.keys():
                i = 0
                distance = 0
                for val in state:
                    distance += (val - s[i])**2
                if distance < best_diff or best_state == None:
                    best_diff = distance
                    best_state = state
            
            if best_state != None:
                q_states[s] = q_states[best_state]
                action = env.action_space.sample()
            else:
                # Take a random actions if you've never felt like this before.
                action = env.action_space.sample()

                # Initialize this new state.
                q_states[s] = {}

                for act in xrange(env.action_space.n):
                    q_states[s][act] = 1
        
        return action

    for i_episode in range(5000):
        # Reset and note initial state.
        observation = env.reset()

        # State is represented with binned states independent of the environment.
        state = state_from_observation(observation)
        
        
        # Run 200 time steps
        for t in range(200):
            # Save the previous state.
            prev_state = state
            
            if i_episode % 500 == 0:
                env.render()
            
            # Pick an action and step forward.
            next_action = get_action(prev_state, env)
            
            observation, reward, done, info = env.step(next_action)
            if len(info) > 0:
                print info

            # Format the new state.
            state = state_from_observation(observation)

            # Reward agents that meet the threshhold for success, punish others.      
            if done:
                reward = t + 1 - goal_score

            # Update Q states.
            q_func(prev_state, next_action, reward, state)
            
            # Check it it's done enough.
            if done or t == 199:
                # Track average scores each 100 trials.
                avg_score += (t + 1)

                #alpha *= 0.999
                # Log successes
                if t + 1 >= goal_score:
                    good_results = good_results + 1
                    # Every success reduced the chance of random actions.
                    #random_act *= 0.9
                
                # Print some debugging/logging information.
                if (i_episode + 1) % 100 == 0:
                    print '{0} average score at {1}, {2} states'.format(float(avg_score) / 100.0, i_episode + 1, len(q_states))
                    
                    # Reset per 100 variables.
                    avg_score = 0
                    good_results = 0

                break
            
    env.monitor.close()

if __name__ == "__main__":
    main()