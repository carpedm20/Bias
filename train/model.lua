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
  self.decoder = self:new_decoder()

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
function Model:new_decoder()
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
        local inx = indices[i + j - 1]
        local x = data.x[idx]
        local y = data.y[idx]

        -- self.emb = nn.LookupTableGPU(config.emb_vecs(1), self.emb_dim)
        local inputs = self.emb:forward(x)
        local rep
        if self.structure =='lstm' then
          rep. self.lstm:forward(inputs)
        elseif self.structure == 'bilstm' then
          rep = {
            self.lstm:forward(inputs)
            self.lstm_b:forward(inputs, true),
          }
        end

        local output = self.decoder
end
