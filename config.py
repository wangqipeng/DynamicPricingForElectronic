#import torch

#DEVICE = torch.device('cuda')
SCREEN_WIDTH = 600
TARGET_UPDATE = 10
EPOCHS = 500
BATCH_SIZE = 128
GAMMA = 0.999
ALPHA = 0.1
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 200
LEARNING_RATE = 0.01
EPSILON_MIN = 0.05
EPSILON = 0.9