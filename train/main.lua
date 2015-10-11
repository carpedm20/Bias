require('init')

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

model_name = 'Bidirectional LSTM'
model_class = LSTM

local data_dir = 'data/'
local vocab = Vocab(data_dir .. 'vocab.txt')

print('loading word embeddings')
local emb=dir = 'data/glove/'
local emb_prefix
