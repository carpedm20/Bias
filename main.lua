require('train/init')

tokenizer = require 'libs/tokenizer'

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

model_name = 'Bidirectional LSTM'
model_class = LSTM

config = {nclusters = 1,
          threshold = 0,
          data_path = 'data',
          name = 'test',
          eos = true,
          char_mode = true}

config.in_f = paths.concat(config.data_path, config.name)..".txt"
config.out_f = paths.concat(config.data_path, config.name)..".tokenized.txt"

dictfname = config.name .. '.dictionary' ..
            '_nclust=' .. config.nclusters ..
            '_thresh=' .. config.threshold ..
            '_charmode=' .. tostring(config.char_mode) ..
            '.th7'

force = true
dic_find = false
for f in paths.files(config.data_path) do
  if string.find(f, dictfname) then
    dict = torch.load(paths.concat(config.data_path, f))
    dic_find = true
  end
end

if (dic_find and not force) == false then
  print(" [#] Build new dictionary")
  dict = tokenizer.build_dictionary(config, config.in_f)
else
  print(" [#] Load existing dictionary")
end

print(" [#] Dictionary size : " .. dict.index_to_freq:size(1))

x, y = tokenizer.tokenize(dict, config.in_f, config.out_f, config, false, true)

local test_length = 20

function test_x_y(x, y, dict, test_length, vertical)
  local vertical = vertical or true
  if vertical then
    for i=1, test_length do
      printf("%s:%s\n",dict.index_to_symbol[x[i]], y[i])
    end
  else
    for i=1, test_length do
      printf("%s,",dict.index_to_symbol[x[i]])
    end
    printf("\n")
    for i=1,test_length do
      printf("%s,",y[i])
    end
    printf("\n")
  end
end

test_x_y(x,y,dict,30)
