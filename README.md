# Proximal Policy Optimization

Implementation of https://arxiv.org/abs/1707.06347 from OpenAI for continuous action spaces. 

This implementation is designed to be **simple** and **easy to read**. No complicated logic or unnecessary Python magic. 

Based on the code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr. 

# Usage
`python train.py` to train. 

`python evaluate.py` to test.

## Requirements
All you need are PyTorch, Gym, and possibly MuJoCo depending on what environment you want to run. For training you will also need [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) installed. 

