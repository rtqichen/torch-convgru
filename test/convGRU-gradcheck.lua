require 'torch'
require 'nn'

local gradcheck = require 'util.gradcheck'
require 'SpatialConvGRU'

local tests = torch.TestSuite()
local tester = torch.Tester()


local function check_size(x, dims)
  tester:asserteq(x:dim(), #dims)
  for i, d in ipairs(dims) do
    tester:assert(x:size(i) == d)
  end
end

function gradCheckTestFactory(N, T, H, W, D, F, scale, dtype)
  dtype = dtype or 'torch.DoubleTensor'
  return function()
    local x = torch.randn(N, T, D, H, W)
    local h0 = torch.randn(N, F, H/scale, W/scale)

    local rnn = nn.SpatialConvGRU(D, F, 3,3, scale)
    local h = rnn:forward{h0, x}

    local dh = torch.randn(#h)

    rnn:zeroGradParameters()
    local weights, gradients = rnn:getParameters()
    local dh0, dx = unpack(rnn:backward({h0, x}, dh))
    local dw = gradients:clone()

    local function fx(x)   return rnn:forward{h0, x} end
    local function fh0(h0) return rnn:forward{h0, x} end

    local function fw(w)
      local old_w = weights
      weights[{}] = w
      local out = rnn:forward{h0, x}
      weights[{}] = old_w
      return out
    end

    local dx_num = gradcheck.numeric_gradient(fx, x, dh)
    local dh0_num = gradcheck.numeric_gradient(fh0, h0, dh)
    local dw_num = gradcheck.numeric_gradient(fw, weights, dh)

    local dx_error = gradcheck.relative_error(dx_num, dx)
    local dh0_error = gradcheck.relative_error(dh0_num, dh0)
    local dw_error = gradcheck.relative_error(dw_num, dw)

    tester:assert(dx_error < 1e-3)
    tester:assert(dh0_error < 1e-3)
    tester:assert(dw_error < 1e-3)
  end
end

tests.gradCheckTest = gradCheckTestFactory(2, 3, 4, 4, 4, 5, 2)

--[[
Check that everything works when we don't pass an initial hidden state.
By default this should zero the hidden state on each forward pass.
--]]
function tests.noInitialStateTest()
  local N, T, D, F = 5, 6, 7, 8
  local H, W = 24, 24
  local scale = 2
  local rnn = nn.SpatialConvGRU(D, F, 3,3, scale)
  
  -- Run multiple forward passes to make sure the state is zero'd each time
  for t = 1, 3 do
    local x = torch.randn(N, T, D, H, W)
    local dout = torch.randn(N, T, F, H/scale, W/scale)

    local out = rnn:forward(x)
    tester:assert(torch.isTensor(out))
    check_size(out, {N, T, F, H/scale, W/scale})

    local din = rnn:backward(x, dout)
    tester:assert(torch.isTensor(din))
    check_size(din, {N, T, D, H, W})

    tester:assert(rnn.h0:sum() == 0)
  end
end


--[[
If we set rnn.remember_states then the initial hidden state will the the
final hidden state from the previous forward pass. Make sure this works!
--]]
function tests.rememberStateTest()
  local N, T, D, F = 5, 6, 7, 8
  local H, W = 24, 24
  local scale = 2
  local rnn = nn.SpatialConvGRU(D, F, 3,3, scale)
  rnn.remember_states = true

  local final_h
  for t = 1, 3 do
    local x = torch.randn(N, T, D, H, W)
    local dout = torch.randn(N, T, F, H/scale, W/scale)

    local out = rnn:forward(x)
    local din = rnn:backward(x, dout)
    if t > 1 then
      tester:assertTensorEq(final_h, rnn.h0, 0)
    end
    final_h = out[{{}, T}]:clone()
  end

  -- After calling resetStates() the initial hidden state should be zero
  rnn:resetStates()
  local x = torch.randn(N, T, D, H, W)
  local dout = torch.randn(N, T, F, H/scale, W/scale)
  rnn:forward(x)
  rnn:backward(x, dout)
  tester:assertTensorEq(rnn.h0, torch.zeros(N, F, H/scale, W/scale), 0)
end


tester:add(tests)
tester:run()