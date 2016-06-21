import gym

filename = 'tmp/cartpole-experiment'
key_file = open('api.key', 'r')
gym_key = key_file.readline()
gym.upload(filename, api_key=gym_key, algorithm_id='alg_QlBKNWQGi2ffX4cMlUw')