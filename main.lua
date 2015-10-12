require('train/init')

tokenizer = require 'libs/tokenizer'

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

model_name = 'Bidirectional LSTM'
model_class = LSTM

config = {nclusters = 1,
          threshold = 0,
          dest_path = 'data',
          name = 'cast',
          char_mode = true}
tokens = tokenizer.build_dictionary(config, 'data/cast.txt')
