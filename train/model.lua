require('nn')

local Model = torch.class('Model')

function Model:__init(config)
  self.mem_dim = config.mem_dim or 120
  self.learning_rate = config.learning_rate = 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.1
  self.num_layers = config.num_layers or 1
  self.batch_size = config.batch_size or 5
  self.reg = config.reg or 1e-4
  self.structure = config.structure or 'lstm'
  self.dropout = (config.dropout == nil) and true or config.dropout
  self.tensortype = torch.getdefaulttensortype()
  self.decoder_option = (config.decoder_option == 'all') and 'all' or 'last'

  -- emb_vecs = [word_size x embed_dim]
  self.emb_dim = config.emb_vecs:size(2) or 1
  -- nn.LookupTable(Size of dictionary, Size of embeding (output) dimension)
  -- self.emb = nn.LookupTable(config.emb_vecs(1), self.emb_dim)
  self.emb = nn.LookupTableGPU(config.emb_vecs(1), self.emb_dim)
  self.emb.weight:copy(config.emb_vecs)

  self.in_zeros = torch.zeros(self.emb_dim)
  self.num_classes = 2

  self.optim_state = {learningRate = self.learning_rate}
  self.criterion = nn.ClassNLLCriterion()

  if self.decoder_option == 'all' then
    self.decoder = self:new_decoder1()
  else
    self.decoder = self:new_decoder2()
  end

  local lstm_config = {
    input_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
    gate_output = true,
  }

  if self.structure == 'lstm' then
    self.lstm = LSTM(lstm_config)
  elseif self.structure == 'bilstm' then
    self.lstm = LSTM(lstm_config)
    self.lstm_b = LSTM(lstm_config)
  else
    error('Wrong structure: '..self.structure)
  end

  -- This Parallel model is not used during forward and backward pass
  -- It is just a model to extract parameters
  local encoder = nn.Parallel()
    :add(self.lstm)
    :add(self.decoder)
  self.params, self.grad_params = modules:gradParameters()

  if self.structure == 'bilstm' then
    share_params(self.lstm_b, self.lstm)
  end
end

-- predict the sentiment of a phrase using the representation
-- given by the final LSTM hidden state
function Model:new_decoder1()
  local input_dim = self.num_layers * self.mem_dim
  local inputs, dec
  if self.structure == 'lstm' then
    local rep = nn.Identity()()
    if self.num_layers == 1 then
      dec = {rep}
    else
      dec = nn.JoinTalbe(1)(rep)
    end
    inputs = {rep}
  elseif self.structure == 'bilstm' then
    local frep, brep = nn.Identity()(), nn.Identity()()
    input = input_dim * 2
    if self.num_layers == 1 then
      dec = nn.JoinTable(1){frep, brep}
    else
      dec = nn.JoinTable(1){nn.JoinTalbe(1)(frep), nn.JoinTable(1)(brep)}
    end
    inputs = {frep, brep}
  end

  local logprobs
  if self.dropout then
    logprobs = nn.LogSoftMax()(
      nn.Linear(input_dim, self.num_classes)(
        nn.Dropout()(dec)))
  else
    logprobs = nn.LogSoftMax()(
      nn.Linear(input_dim, self.num_classes))
  end

  return nn.gModule(inputs, {logprobs})
end

function Model:new_decoder2()
  local input_dim = 1 * self.mem_dim
  local inputs, dec
  if self.structure == 'lstm' then
    local rep = nn.Identity()()
    if self.num_layers == 1 then
      dec = {rep}
    else
      dec = rep[1]
    end
    inputs = {rep}
  elseif self.structure == 'bilstm' then
    local frep, brep = nn.Identity()(), nn.Identity()()
    input = input_dim * 2
    if self.num_layers == 1 then
      dec = nn.JoinTable(1){frep, brep}
    else
      dec = nn.JoinTable(1){frep[1], brep[1]}
    end
    inputs = {frep, brep}
  end

  local logprobs
  if self.dropout then
    logprobs = nn.LogSoftMax()(
      nn.Linear(input_dim, self.num_classes)(
        nn.Dropout()(dec)))
  else
    logprobs = nn.LogSoftMax()(
      nn.Linear(input_dim, self.num_classes))
  end

  return nn.gModule(inputs, {logprobs})
end

function Model:train(dataset)
  -- nn:training() : This sets the mode of the Module (or sub-modules) to train=true. This is useful for modules like Dropout that have a different behaviour during training vs evaluation.
  -- nn:evaluate() : This sets the mode of the Module (or sub-modules) to train=false.
  self.lstm:training()
  self.decoder:training()
  if self.structure == 'bilstm' then
    self.lstm_b:training()
  end

  local indicies = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  for i=1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i+self.batch_size-1, dataset.size)-i+1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0
      for j=1, batch_size do
        local idx = indices[i + j - 1]
        local x = data.x[{{idx, idx+self.window_size}}]
        local y = data.y[{{idx, idx+self.window_size}}]

        -- self.emb = nn.LookupTableGPU(config.emb_vecs(1), self.emb_dim)
        local inputs = self.emb:forward(x)
        local rep
        -- self.output = {}
        -- for i = 1, self.num_layers do
        --   self.output[i] = htable[i]
        -- end
        -- return self.output
        if self.structure =='lstm' then
          rep = self.lstm:forward(inputs)
        elseif self.structure == 'bilstm' then
          rep = {
            self.lstm:forward(inputs)
            self.lstm_b:forward(inputs, true),
          }
        end

        local output = self.decoder:forward(rep)
        local example_loss = self.criterion:forward(output, y)
        loss = loss + example_loss
        local obj_grad = self.criterion:backward(output, y)
        local rep_grad = self.decoder:backward(rep, obj_grad)
        local input_grads
        if self.structure == 'lstm' then
          input_grads = self.LSTM_backward(x, inputs, rep_grad)
        elseif self.structure == 'bilstm' then
          input_grads = self.LSTM_backward(x, inputs, rep_grad)
        end
        self.emb:backward(span, input_grads)
      end
    
      loss = loss / batch_size
      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)

      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params) -- ???
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
    self.emb:updateParameters(self.emb_learning_rate)
  end
  xlua.progress(dataset.size, dataset.size)
end

function Model:LSTM_backward(x, inputs, rep_grad)
  local grad
  if self.num_layers == 1 then
    gard = torch.zeros(x:nElement(), self.mem_dim)
    grad[x:nElement()] = rep_grad
  else
    grad = torch.zeros(x:nElement(), self.num_layers, self.mem_dim)
    for l=1, self.num_layers do
      grad[{x, l, {}}] = rep_gard[l]
    end
  end
  local input_grads = self.lstm:backward(inputs,grad)
  return input_grads
end

function Model:LSTM_backward(x, inputs, rep_grad)
  local grad
  if self.num_layers == 1 then
    gard = torch.zeros(x:nElement(), self.mem_dim)
    grad[x:nElement()] = rep_grad
  else
    grad = torch.zeros(x:nElement(), self.num_layers, self.mem_dim)
    for l=1, self.num_layers do
      grad[{x, l, {}}] = rep_gard[l]
    end
  end
  local input_grads = self.lstm:backward(inputs,grad)
  return input_grads
end
