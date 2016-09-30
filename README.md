## torch-convgru
Dumping code for a convolutional GRU network implementation using nngraph. Allows "parameter sharing relaxation".

Modification of torch-rnn's code base: https://github.com/jcjohnson/torch-rnn
Uses nngraph library to compute step-wise outputs and gradients for easy extendability.

#### Gated Recurrent Unit with Convolutional Mappings

```
This module takes as input a tensor of size (N,T,H,W,D)
and outputs a tensor of size (N,T,H,W,H)
Where N is the minibatch size. 
T is the length of the time domain.
H and W are height and width of an image, not constrained.
D is the number of input features.
F is the number of hidden features.
For now, only works if kernel width and height are odd,
and forces a stride of 1.
```

#### "Parameter Sharing Relaxation" ( http://arxiv.org/abs/1511.08228 )
Instead of sharing weights at every time step, weights are shared at every `r`-th time step.

ie. at time step `t`, the module uses the `i`-th set of weights where `i = t mod r.`

Weights are penalized by `p*| Avg - Weight |` where `Avg` is the average of the weights and `p` is a penalty coefficient.


#### Usage
```
module = nn.SpatialConvGRU(input_dim, hidden_dim, kW, kH, [hidden_downscale], [relaxation], [relaxationPenality])
```

- `hidden_downscale` (integer > 0) Lets the hidden state have a downsampled size wrt. the input. Defaults to `1`.
- `relaxation` (integer > 0) The number of relaxed time steps. Defaults to `1`.
