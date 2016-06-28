import random
import gym

Random_Count = 0
Q_States = {}

def main(argv=None):
    global Random_Count
    # Initial values and logging.
    alpha = 0.6
    gamma = 0.25
    random_act = 0.5
    avg_score = 0
    good_results = 0
    goal_score = 195
    env = gym.make('CartPole-v0')
    filename = 'tmp/cartpole-experiment'
    env.monitor.start(filename, force=True)

    # Run the Q function to update states.
    def q_func(s, a, r, s_p):
        global Q_States
        if s_p not in Q_States.keys():
            Q_States[s_p] = {}
            for act in xrange(env.action_space.n):
                Q_States[s_p][act] = 1
        
        next_best = max(Q_States[s_p].values())
        
        Q_States[s][a] = (1.0 - alpha) * Q_States[s][a] + alpha * (r + gamma * next_best)
        
    # Get the next action.
    def get_action(s, env):
        global Q_States
        global Random_Count
    
        if s in Q_States.keys():
            # default action is random if nothing is better.
            best = -10
            best_act = env.action_space.sample()
            for k, v in Q_States[s].iteritems():
                if v > best:
                    best = v
                    best_act = k

            action = best_act
        else:
            # Take a random actions if you've never felt like this before.
            action = env.action_space.sample()

            # Initialize this new state.
            Q_States[s] = {}

            for act in xrange(env.action_space.n):
                Q_States[s][act] = 1

        # Occasionally randomness to break it out of a funk.
        if random.random() < random_act:
            action = env.action_space.sample()
            Random_Count += 1
        return action

    for i_episode in range(5000):
        # Reset and note initial state.
        observation = env.reset()

        # State is represented with binned states independent of the environment.
        state = ()
        for val in observation:
            temp_value = round(val * 4.0)/4.0
            if temp_value == -0.0:
                temp_value = 0.0
            state += (temp_value, )
        
        
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
            state = ()
            for val in observation:
                temp_value = round(val * 4.0)/4.0
                if temp_value == -0.0:
                    temp_value = 0.0
                state += (temp_value, )

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
                    print '{0} average score at {1}, {2}'.format(float(avg_score) / 100.0, i_episode + 1, Random_Count)
                    
                    # Reduce randomness every 100 trials.
                    random_act *= 0.8

                    # Reset per 100 variables.
                    Random_Count = 0
                    avg_score = 0
                    good_results = 0

                break
            
    env.monitor.close()

if __name__ == "__main__":
    main()