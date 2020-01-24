import DQN_model as agent
import gym
class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        env.reset()
    def train(self, agent):
        timestep = 0
        for i in range(0, 10000):
            state = env.reset()
            count = 0
            while True:
                #print("done")
                timestep = timestep + 1
                count = count + 1
                env.render();
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.add_to_replay((state, action, reward, next_state, done))
                if done:
                    print(count)
                    count = 0
                    env.reset()
                    break
                if timestep % 1000 == 0:
                    agent.update_target()
                agent.update_model()
                state = next_state





if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    CartPole_env   = CartPole()
    CartPole_agent = agent.Agent(env.action_space.n, 4)
    CartPole_env.train(CartPole_agent)
    print(env.action_space.n)
    env.reset()

    env.close()
