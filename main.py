import sys
sys.path.append('../gym')
import gym
# class CartPole:
#     def __init__():
#         self.env = gym.make('CartPole-v1')
#         env.reset()
#     def train():
#
#         for i in range(0, 10000):
#             env.render();
#             oberervation, reward, done, info = env.step(env.action_space.sample())
#             if done:
#                 env.reset()
#
#

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    print(env.action_space.n)
    print(env.observation_space[0])
    env.reset()

    env.close()
