local LSTM, parent = torch.class('LSTM', 'nn.Module')

function LSTM:__init(config)
  parent.__init(self)

  self.input_dim = config.input_dim or 100
  self.mem_dim = config.mem_dim or 150
  self.num_layers = config.num_layers or 1
  self.gate_output = config.gate_output or true
  if self.gate_output == nil then self.gate_output = true end

  self.master_cell = self:new_cell()
  self.depth = 0
  self.cells = {} -- roll out

  local ctable_init, ctable_grad, htable_init, htable_grad
  if self.num_layers == 1 then
    ctable_init = torch.zeros(self.mem_dim)
    htable_init = torch.zeros(self.mem_dim)
    ctable_grad = torch.zeros(self.mem_dim)
    htable_grad = torch.zeros(self.mem_dim)
  else
    ctable_init, ctable_grad, htable_init, htable_grad = {}, {}, {}, {}
    for i = 1, self.num_layers do
      ctable_init[i] = torch.zeros(self.mem_dim)
      htable_init[i] = torch.zeros(self.mem_dim)
      ctable_grad[i] = torch.zeros(self.mem_dim)
      htable_grad[i] = torch.zeros(self.mem_dim)
    end
  end
  self.initial_values = {ctable_init, htable_init}
  self.gradInput = {
    torch.zeros(self.input_dim),
    ctable_grad,
    htable_grad
  }
end

function LSTM:new_cell()
  local input, ctable_prev, htable_prev = nn.Identity()(), nn.Identity()(), nn.Identity()()

  local htable, ctable = {}, {}
  for layer = 1, self.num_layers do
    local h_prev = (self.num_layers == 1) and htable_prev or nn.SelectTable(layer)(htable_prev)
    local c_prev = (self.num_layers == 1) and ctable_prev or nn.SelectTable(layer)(ctable_prev)

    local new_gate = function()
      local in_module = (layer == 1)
        and nn.Linear(self.input_dim, self.mem_dim)(input)
        or nn.Linear(self.mem_dim, self.mem_dim)(htable[layer - 1])
      -- W_{xi} x_t + W_{hi} H_{t-1}
      return nn.CAddTable(){
        in_module,
        nn.Linear(self.mem_dim, self.mem_dim)(h_prev)
      }
    end

    -- logistic(W_{xi} x_t + W_{hi} H_{t-1})
    local i = nn.Sigmoid()(new_gate())
    local f = nn.Sigmoid()(new_gate())
    -- th(W_{xg} x_t + W_{hg} H_{t-1})
    local u = nn.Tanh()(new_gate())

    -- c_t = f .* c_{t-1} + i .* g
    ctable[layer] = nn.CAddTable(){
      nn.CMulTable(){f, c_prev},
      nn.CMulTable(){i, update}
    }

    -- h_t = o .* t(c_t)
    if self.gate_output then
      local o = nn.Sigmoid()(new_gate())
      htable[layer] = nn.CMulTable(){o, nn.Tanh()(ctable[layer])}
    else
      htable[layer] = nn.Tanh()(ctable[layer])
    end
  end

  htable, ctable = nn.Identity()(htable), nn.Identity()(ctable)
  local cell = nn.gModule({input, ctable_prev, htable_prev}, {ctable, htable})

  if self.master_cell then
    shar_params(cell, self.master_cell)
  end
  return cell
end

-- inputs: T x input_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left
function LSTM:forward(inputs, reverse)
  local size = inputs:size(1)
  for t = 1, size do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    self.depth = self.depth + 1
    local cell = self.cells[self.depth]
    if cell == nil then
      cell = self:new_cell()
      self.cells[self.depth] = cell
    end
    local prev_output
    if self.depth > 1 then
      prev_output = self.cells[self.depth - 1].output
    else
      prev_output = self.initial_values
    end

    local outputs = cell:forward({input, prev_output[1], prev_output[2]})
    local ctable, htable = unpack(outputs) -- ???
    if self.num_layers == 1 then
      self.output = htable
    else
      self.output = {}
      for i=1, self.num_layers do
        self.ouput[i] = htable[i]
      end
    end
  end
  return self.output
end

-- inputs: T x input_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- reverse: if true, read the input from right to left.
function LSTM:backward(inputs, grad_outputs, reverse)
  local size = inputs:size(1)
  if self.depth == 0 then
    error("No cells to backpropagate")
  end

  local input_grads = torch.Tensor(inputs:size())
  for t=size, 1, -1 do
    local input = reverse and inputs[size-t+1] or inputs[t]
    local grad_output = reverse and grad_outputs[size-t+1] or grad_outputs[t]
    local cell = self.cells[self.depth]
    -- self.gradInput = {
    --   torch.zeros(self.input_dim),
    --   ctable_grad,
    --   htable_grad -- list of htable_grad for each layer with zero initialization
    -- }
    local grads = {self.gradInput[2], self.gradInput[3]}
    if self.num_layers == 1 then
      grads[2]:add(grad_output)
    else
      for i=1, self.num_layers do
        grads[2][i]:add(grad_output[i])
      end
    end

    local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
      or self.initial_values -- self.initial_values = {ctable_init, htable_init}

    self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)
    if reverse then
      input_grads[size-t+1] = self.gradInput[1] -- torch.zeros(self.input_dim)
    else
      input_grads[t] = self.gradInput[1]
    end
    self.depth = self.depth - 1
  end
  self:forget()

  return input_grads -- T x input_dim tensor which includes gradients
end

function LSTM:share(lstm, ...)
  if self.in_dim ~= lstm.in_dim then error("LSTM input dimension mismatch") end
  if self.mem_dim ~= lstm.mem_dim then error("LSTM memory dimension mismatch") end
  if self.num_layers ~= lstm.num_layers then error("LSTM layer count mismatch") end
  if self.gate_output ~= lstm.gate_output then error("LSTM output gating mismatch") end
  share_params(self.master_cell, lstm.master_cell, ...)
end

function LSTM:zeroGradParameters()
  self.master_cell:zeroGradParameters()
end

function LSTM:parameters()
  return self.master_cell:parameters()
end

function LSTM:forget()
  -- self.gradInput = {
  --   torch.zeros(self.input_dim),
  --   ctable_grad,
  --   htable_grad -- list of htable_grad for each layer with zero initialization
  -- }
  self.depth = 0
  for i = 1, #self.gradInput do
    local gradInput = self.gradInput[i]
    if type(gradInput) == 'table' then
      for _, t in pairs(gradInput) do t:zero() end
    else
      self.gradInput[i]:zero()
    end
  end
end

function LSTM:clipGradParams(gclip)
  for k,v in pairs(self.nets) do
    local lw, ldw = v:parameters()
    if ldw then
      for i = 1, #ldw do
        if ldw[i] then
          self.clip_function(ldw[i], gclip)
        end
      end
    end
  end
end
