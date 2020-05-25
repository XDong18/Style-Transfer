import torch
import torch.nn.functional as F


def content_loss(input, target):
    _input = input[0]
    _target = target[0]
    return F.mse_loss(_input, _target)

def gram_matrix(inputs):
    a, b, c, d = inputs.size()
    features = inputs.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def style_loss(input, target):
    num = len(input)
    loss = 0
    for i_input, i_target in zip(input, target):
        loss += F.mse_loss(gram_matrix(i_input), gram_matrix(i_target))
    
    return loss / num
    
