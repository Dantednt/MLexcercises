import math

def sigmoid(x):
    y = 1.0 / (1 + math.exp(-x))
    return y


def activate(inputs, weights):
    # perform net input
    h = 0
    for x, w in zip(inputs, weights): 
        h += x*w

    # perfom activation
    return sigmoid(h)


if __name__ == '__main__':
  inputs = [.5, .3, .2]
  weights = [.4, .7, .2]
  outputs = activate(inputs, weights)
  print(outputs)