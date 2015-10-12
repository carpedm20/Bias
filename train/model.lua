require('nn')
require('fbcunn')

local Model = torch.class('Model')

function Model:__init(config)
  self.window_size = config.window_size or 55555
  self.mem_dim = config.mem_dim or 120
  self.learning_rate = config.learning_rate or 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.1
  self.num_layers = config.num_layers or 1
  self.batch_size = config.batch_size or 5
  self.reg = config.reg or 1e-4
  self.structure = config.structure or 'lstm'
  self.dropout = (config.dropout == nil) and true or config.dropout
  self.tensortype = torch.getdefaulttensortype()
  self.decoder_option = (config.decoder_option == 'all') and 'all' or 'last'

  self.emb_dim = 100
  -- nn.LookupTable(Size of dictionary, Size of embeding (output) dimension)
  self.emb = nn.LookupTableGPU(config.dict.index_to_freq:size(1), self.emb_dim)

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
  self.params, self.grad_params = encoder:getParameters()

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

function Model:train(data)
  -- nn:training() : This sets the mode of the Module (or sub-modules) to train=true. This is useful for modules like Dropout that have a different behaviour during training vs evaluation.
  -- nn:evaluate() : This sets the mode of the Module (or sub-modules) to train=false.
  assert(x:size(1) == y:size(1), "Dimensions of x (" .. x:size(1) .. ") and y (" .. y:size(1) .. ")  are different")

  self.lstm:training()
  self.decoder:training()
  if self.structure == 'bilstm' then
    self.lstm_b:training()
  end

  local indices = torch.randperm(data.y:size(1) - self.window_size + 1)
  local zeros = torch.zeros(self.mem_dim)
  for i=1, data.x:size(1), self.batch_size do
    xlua.progress(i, data.x:size(1))
    local batch_size = math.min(i+self.batch_size-1, data.x:size(1))-i+1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0
      for j=1, batch_size do
        local idx = indices[i + j - 1]
        local x = data.x[{{idx, idx+self.window_size}}]
        local y = data.y[{{idx, idx+self.window_size}}]

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
            self.lstm:forward(inputs),
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
  xlua.progress(data.x:size(1), data.x:size(1))
end

function Model:LSTM_backward(sent, inputs, rep_grad)
  local grad
  if self.num_layers == 1 then
    grad = torch.zeros(sent:nElement(), self.mem_dim)
    grad[sent:nElement()] = rep_grad
  else
    grad = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
    for l = 1, self.num_layers do
      grad[{sent:nElement(), l, {}}] = rep_grad[l]
    end
  end
  local input_grads = self.lstm:backward(inputs, grad)
  return input_grads
end

-- Bidirectional LSTM backward propagation
function Model:BiLSTM_backward(sent, inputs, rep_grad)
  local grad, grad_b
  if self.num_layers == 1 then
    grad   = torch.zeros(sent:nElement(), self.mem_dim)
    grad_b = torch.zeros(sent:nElement(), self.mem_dim)
    grad[sent:nElement()] = rep_grad[1]
    grad_b[1] = rep_grad[2]
  else
    grad   = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
    grad_b = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
    for l = 1, self.num_layers do
      grad[{sent:nElement(), l, {}}] = rep_grad[1][l]
      grad_b[{1, l, {}}] = rep_grad[2][l]
    end
  end
  local input_grads = self.lstm:backward(inputs, grad)
  local input_grads_b = self.lstm_b:backward(inputs, grad_b, true)
  return input_grads + input_grads_b
end

function Model:predict(x)
  self.lstm:evaluate()
  self.decoder:evaluate()
  local inputs = self.emb:forward(x)

  local rep
  if self.structure == 'lstm' then
    rep = self.lstm:forward(inputs)
  elseif self.structure == 'bilstm' then
    self.lstm_b:evaluate()
    rep = {
      self.lstm:forward(inputs),
      self.lstm_b:forward(inputs, true),
    }
  end
  local logprobs = self.decoder:forward(rep)
  local prediction
  -- if self.fine_grained then
  prediction = argmax(logprobs)
  -- else
  --   prediction = (logprobs[1] > logprobs[3]) and 1 or 3
  -- end
  self.lstm:forget()
  if self.structure == 'bilstm' then
    self.lstm_b:forget()
  end
  return prediction
end

-- Produce sentiment predictions for each sentence in the dataset.
function Model:predict_dataset(data)
  local predictions = torch.Tensor(data.x:size(1))
  for i = 1, data.x:size(1) do
    xlua.progress(i, data.x:size(1))
    local x = data.x[{{idx, idx+self.window_size}}]
    predictions[i] = self:predict(x)
  end
  return predictions
end

function argmax(v)
  local idx = 1
  local max = v[1]
  for i = 2, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

function Model:print_config()
  local num_params = self.params:size(1)
  local num_decoder_params = self:new_decoder1():getParameters():size(1)
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_decoder_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'window size', self.window_size)
  printf('%-25s = %d\n',   'LSTM memory dim', self.mem_dim)
  printf('%-25s = %s\n',   'LSTM structure', self.structure)
  printf('%-25s = %d\n',   'LSTM layers', self.num_layers)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %s\n',   'dropout', tostring(self.dropout))
end

function Model:save(path)
  local config = {
    batch_size        = self.batch_size,
    dropout           = self.dropout,
    emb_learning_rate = self.emb_learning_rate,
    emb_vecs          = self.emb.weight:float(),
    learning_rate     = self.learning_rate,
    num_layers        = self.num_layers,
    mem_dim           = self.mem_dim,
    reg               = self.reg,
    structure         = self.structure,
    window_size       = self.window_size,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function Model.load(path)
  local state = torch.load(path)
  local model = treelstm.Model.new(state.config)
  model.params:copy(state.params)
  return model
end
