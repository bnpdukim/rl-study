import gym
import numpy as np

env = gym.make("CartPole-v0")
env.reset()
img = env.render(mode="rgb_array")
print(img.shape)
print(env.action_space)

action = 1
observation, reward, done, info = env.step(action)
print("관측값 : ", observation) # cart position, cart velocity, pole angle, pole velocity at top
print("보상 : ",reward)
print("에피소드 진행여부 : ",done)
print("디버깅 정보 : ", info)

def basic_policy(obs):
  poleAngle = obs[2]
  return 0 if poleAngle < 0 else 1

totals = []

for episode in range(500):
  episode_rewards = 0
  obs = env.reset()
  for step in range(1000):
    action = basic_policy(obs)
    obs, reward, done, info = env.step(action)
    episode_rewards += reward
    if done:
      break
  else:
    print("episode(", episode, ") success, 1000 loop complete")
  totals.append(episode_rewards)

print("mean : ", np.mean(totals), ", std : ", np.std(totals), ", min : ", np.min(totals), ", max : ",np.max(totals))

env.close()
