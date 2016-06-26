require 'torch'
require 'nn'
require 'nngraph'

local layer, parent = torch.class('nn.SpatialConvGRU', 'nn.Module')

--[[
Modification of torch-rnn's code base.
https://github.com/jcjohnson/torch-rnn

Currently just uses nn library to compute step-wise
outputs and gradients. Extra computation and memory
at the expense of easy extendability.

--- Gated Recurrent Unit with Convolutional Mappings ---

This module takes as input a tensor of size (N,T,H,W,D)
and outputs a tensor of size (N,T,H,W,H)

Where N is the minibatch size. 
T is the length of the time domain.
H and W are height and width of an image, not constrained.
D is the number of input features.
F is the number of hidden features.

For now, only works if kernel width and height are odd,
and forces a stride of 1.
TODO: relax this.

"Parameter Sharing Relaxation" ( http://arxiv.org/abs/1511.08228 )
Instead of sharing weights at every time step, weights are shared
at every r'th time step.
ie. at time step t, the module uses the i'th set of weights
where i = t mod r.

Weights are penalized by p*| Avg - Weight | where Avg is the
average of the weights and p is a penalty coefficient.
!! This penalty is not yet implemented.
--]]
function layer:__init(input_dim, hidden_dim, kW, kH, hidden_downscale, relaxation, relaxationPenality)
    parent.__init(self)

    local R = relaxation or 1
    local coefR = relaxationPenality or 0
    local D, F = input_dim, hidden_dim

    -- check arguments
    if R < 1 then
        error('Relaxation parameter must be a natural number.')
    elseif coefR < 0 then
        error('Relaxation penalty must be non-negative.')
    elseif D < 0 or F < 0 or kW < 0 or kH < 0 then
        error('Convolution parameters must be positive.')
    elseif hidden_downscale > 1 then
        error('hidden_downscale must be less than 1')
    end

    self.scale = 1/hidden_downscale or 1    -- must be a positive integer.
    self.relaxation = R
    self.relaxationPenality = coefR
    self.input_dim, self.hidden_dim = D, F
    self.kW = kW
    self.kH = kH
    self.dW = 1
    self.dH = 1
    self.padW = (kW-1)/2
    self.padH = (kH-1)/2

    self.modules = {}
    self.params = {}
    self.gradParams = {}
    for i=1,R do
        self.modules[i%R] = self:_buildModule()
        self.params[i%R] = self.modules[i%R]:getParameters()
        self.gradParams[i%R] = self.modules[i%R]:getParameters()
    end

    -- TODO: smarter initialization
    -- self:reset()

    self.h0 = torch.Tensor()
    self.remember_states = false

    self.buffer1 = torch.Tensor()
    self.buffer2 = torch.Tensor()
    self.grad_h0 = torch.Tensor()
    self.grad_x = torch.Tensor()
    self.gradInput = {self.grad_h0, self.grad_x}
end

function layer:reset(std)
    if not std then
        std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
    end
    for _, gModule in pairs(self.modules) do
        for _, node in pairs(gModule.forwardnodes) do
            if torch.isTypeOf(node.data.module, 'nn.SpatialConvolution') then
                node.data.module.weight:normal(0,std)
                node.data.module.bias:normal(0,std)
            end
        end
    end
end

function layer:resetStates()
    self.h0 = self.h0.new()
end

--[[
Use nn to construct the recurrent module.
Takes as input {h[t-1],x[t]} and outputs h[t]
where x is (N,D,H,W) and h is (N,F,H,W)
where N is the minibatch size.
The gates and candidate must have the same sizes as h.

--------------------------------------------

ar[t] = conv_r([ x[t] h[t-1] ])
az[t] = conv_z([ x[t] h[t-1] ])

r[t] = sigmoid(ar[t])   # Reset gate
z[t] = sigmoid(az[t])   # Update gate

ac[t] = conv_c([ x[t] r[t]*h[t-1]) ])
c[t] = tanh(ac[t])      # Candidate hidden output

h[t] = (1-z)*c[t] + z[t]*h[t-1]
--]]
function layer:_buildModule()
    local D = self.input_dim
    local F = self.hidden_dim
    local kW, kH, dW, dH, padW, padH = self.kW, self.kH, self.dW, self.dH, self.padW, self.padH

    local inputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    local prev_h = inputs[1]
    local cur_x = inputs[2]

    local combined1 = self.scale == 1 and
    nn.JoinTable(1,3)({cur_x, prev_h}) or
    nn.JoinTable(1,3)({cur_x, nn.SpatialUpSamplingNearest(self.scale)(prev_h)})

    local conv1 = nn.SpatialConvolution(D+F, 2*F, kW, kH, dW, dH, padW, padH)(combined1)

    local gates = self.scale == 1 and 
    nn.Sigmoid()(conv1) or 
    nn.Sigmoid()(nn.SpatialMaxPooling(self.scale, self.scale, self.scale, self.scale)(conv1))

    local reset_gate = nn.Narrow(2,1,F)(gates)        -- these two lines
    local update_gate = nn.Narrow(2,F+1,F)(gates)     -- require first dim to be minibatch dim
    local one_minus_update_gate = nn.MulConstant(-1,true)(nn.AddConstant(-1)(update_gate))

    local reset = nn.CMulTable()({reset_gate, prev_h})
    local update = nn.CMulTable()({update_gate, prev_h})

    local combined2 = self.scale == 1 and
    nn.JoinTable(1,3)({cur_x, reset}) or
    nn.JoinTable(1,3)({cur_x, nn.SpatialUpSamplingNearest(self.scale)(reset)})

    local conv2 = nn.SpatialConvolution(D+F, F, kW, kH, dW, dH, padW, padH)(combined2)

    local candidate = self.scale == 1 and
    nn.Tanh()(conv2) or 
    nn.Tanh()(nn.SpatialMaxPooling(self.scale, self.scale, self.scale, self.scale)(conv2))

    local next_h = nn.CAddTable()({
        nn.CMulTable()({one_minus_update_gate, candidate}),
        update
        })

    -- annotations
    prev_h:annotate{name='h[t-1]'}
    cur_x:annotate{name='x[t]'}
    reset_gate:annotate{name='r[t]'}
    update_gate:annotate{name='z[t]'}
    candidate:annotate{name='c[t]'}
    next_h:annotate{name='h[t]'}

    return nn.gModule(inputs, {next_h})
end

function layer:_unpack_input(input)
    local h0, x = nil, nil
    if torch.type(input) == 'table' and #input == 2 then
        h0, x = unpack(input)
    elseif torch.isTensor(input) then
        x = input
    else
        assert(false, 'invalid input')
    end
    return h0, x
end

local function check_dims(x, dims)
    assert(x:dim() == #dims)
    for i, d in ipairs(dims) do
        assert(x:size(i) == d)
    end
end

function layer:_get_sizes(input, gradOutput)
    local h0, x = self:_unpack_input(input)
    local N, T = x:size(1), x:size(2)
    local H, W = x:size(4), x:size(5)
    local F, D = self.hidden_dim, self.input_dim
    check_dims(x, {N, T, D, H, W})
    if h0 then
        check_dims(h0, {N, F, H/self.scale, W/self.scale})
    end
    if gradOutput then
        check_dims(gradOutput, {N, T, F, H/self.scale, W/self.scale})
    end
    return N, T, H, W, D, F
end

--[[
Input: Table of
- h0: Initial hidden state of shape (N, F, H, W)
- x:  Sequence of inputs, of shape (N, T, D, H, W)
Output:
- h: Sequence of hidden states, of shape (N, T, F, H, W)
--]]
function layer:updateOutput(input)
    self.recompute_backward = true
    local R = self.relaxation
    local h0, x = self:_unpack_input(input)
    local N, T, H, W, D, F = self:_get_sizes(input)
    self._return_grad_h0 = (h0 ~= nil)

    if not h0 then
        h0 = self.h0
        if h0:nElement() == 0 or not self.remember_states then
            h0:resize(N, F, H/self.scale, W/self.scale):zero()
        elseif self.remember_states then
            local prev_N, prev_T = self.output:size(1), self.output:size(2)
            assert(prev_N == N, 'batch sizes must be constant to remember states')
            h0:copy(self.output[{{}, prev_T}])
        end
    end

    self.output:resize(N, T, F, H/self.scale, W/self.scale):zero()
    local prev_h = h0
    for t = 1, T do
        local cur_x = x[{{}, t}]
        self.output[{{}, t}] = self.modules[t%R]:forward({prev_h, cur_x})
        prev_h = self.output[{{}, t}]
    end

    return self.output
end

-- Normally we don't implement backward, and instead just implement
-- updateGradInput and accGradParameters. However for an RNN, separating these
-- two operations would result in quite a bit of repeated code and compute;
-- therefore we'll just implement backward and update gradInput and
-- gradients with respect to parameters at the same time.
function layer:backward(input, gradOutput, scale)
    self.recompute_backward = false
    scale = scale or 1.0
    assert(scale == 1.0, 'scale must be 1')
    local R = self.relaxation
    local N, T, D, H = self:_get_sizes(input, gradOutput)
    local h0, x = self:_unpack_input(input)
    if not h0 then h0 = self.h0 end
    local grad_h = gradOutput

    local grad_h0 = self.grad_h0:resizeAs(h0):zero()
    local grad_x = self.grad_x:resizeAs(x):zero()
    local grad_next_h = self.buffer1:resizeAs(h0):zero()
    for t = T, 1, -1 do
        local cur_x = x[{{}, t}]
        local next_h, prev_h = self.output[{{}, t}], nil
        if t == 1 then
            prev_h = h0
        else
            prev_h = self.output[{{}, t - 1}]
        end
        grad_next_h:add(grad_h[{{}, t}])
        -- should I optimize for computation at the cost of memory,
        -- by storing a clone of the module?
        self.modules[t%R]:forward({prev_h, cur_x})
        grad_next_h[{}], grad_x[{{},t}] = unpack(self.modules[t%R]:backward({prev_h, cur_x}, grad_next_h))
    end
    grad_h0:copy(grad_next_h)

    if self._return_grad_h0 then
        self.gradInput = {self.grad_h0, self.grad_x}
    else
        self.gradInput = self.grad_x
    end

    local coefR = self.relaxationPenality
    if coefR > 0 then
        -- Calculate average weight
        local avg = self.buffer2:resizeAs(self.params[0]):zero()
        for _, params in pairs(self.params) do
            avg:add(params)
        end
        avg:div(R)

        -- Add penalty of coefR * |w - avg|
        -- Should I use L2 loss instead?
        for i=1,R do
            self.gradParams[i%R]:add(torch.sign(self.params[i%R] - avg):mul(coefR))
        end
    end

    return self.gradInput
end

function layer:updateGradInput(input, gradOutput)
    if self.recompute_backward then
        self:backward(input, gradOutput, 1.0)
    end
    return self.gradInput
end


function layer:accGradParameters(input, gradOutput, scale)
    if self.recompute_backward then
        self:backward(input, gradOutput, scale)
    end
end

function layer:parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i=1,#from do
                tinsert(to,from[i])
            end
        else
            table.insert(to,from)
        end
    end
    local w = {}
    local gw = {}
    for _, mod in pairs(self.modules) do
        local mw,mgw = mod:parameters()
        if mw then
            tinsert(w,mw)
            tinsert(gw,mgw)
        end
    end
    return w,gw
end

function layer:training()
    for _, mod in pairs(self.modules) do
        mod:training()
    end
end

function layer:evaluate()
    for _, mod in pairs(self.modules) do
        mod:evaluate()
    end
end

function layer:clearState()
    self.buffer1:set()
    self.buffer2:set()
    self.grad_h0:set()
    self.grad_x:set()
    self.output:set()
end

function layer:__tostring__()
    local s = string.format('%s(%d -> %d (%.2f), %dx%d', torch.type(self),
        self.input_dim, self.hidden_dim, 1/self.scale, self.kW, self.kH)
    if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
        s = s .. string.format(', %d,%d', self.dW, self.dH)
    end
    if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
        s = s .. ', ' .. self.padW .. ',' .. self.padH
    end
    if self.relaxation ~= 1 then
        s = s .. ', ' .. self.relaxation .. ',' .. self.relaxationPenality
    end
    return s .. ')'
end