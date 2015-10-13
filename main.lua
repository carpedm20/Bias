-- debugger = require('fb.debugger')
require('train/init')

tokenizer = require 'libs/tokenizer'

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

function header(s)
  print(string.rep('=', 50))
print("     "..s)
  print(string.rep('=', 50))
end

nngraph.setDebug(true)
local TEST = true
if TEST then
  config = {nclusters = 1,
            threshold = 0,
            data_path = 'data',
            name = 'test',
            eos = true,
            char_mode = true}
else
  config = {nclusters = 1,
            threshold = 50,
            data_path = 'data',
            name = 'cast',
            eos = true,
            char_mode = true}
end

config.in_f = paths.concat(config.data_path, config.name)..".txt"
config.out_f = paths.concat(config.data_path, config.name)..".tokenized.txt"

dictfname = config.name .. '.dictionary' ..
            '_nclust=' .. config.nclusters ..
            '_thresh=' .. config.threshold ..
            '_charmode=' .. tostring(config.char_mode) ..
            '.th7'

force = false
dic_find = false
for f in paths.files(config.data_path) do
  if string.find(f, dictfname) then
    dict = torch.load(paths.concat(config.data_path, f))
    dic_find = true
  end
end

if (dic_find and not force) == false then
  print("===>[#] Build new dictionary")
  dict = tokenizer.build_dictionary(config, config.in_f)
else
  print("===>[#] Load existing dictionary")
end

print("===>[#] Dictionary size : " .. dict.index_to_freq:size(1))

token_find = false
tokenfname = config.name..".tokenized.txt"
for f in paths.files(config.data_path) do
  if string.find(f, tokenfname) then
    x = torch.load(paths.concat(config.data_path, f))
    yfname, _ = paths.concat(config.data_path, f):gsub(".tokenized.",".segmenter.")
    y = torch.load(yfname)
    token_find = true
  end
end
if (token_find and not force) == false then
  print("===>[#] Build new tokenized texts")
  x, y = tokenizer.tokenize(dict, config.in_f, config.out_f, config, false, true)
else
  print("===>[#] Load existing tokenized texts")
end

print("===>[#] Training data size : " .. x:size(1))


local test_length = 20

function test_x_y(x, y, dict, test_length, vertical, start)
  local vertical = vertical or true
  local start = start or 0
  if vertical then
    for i=start+1, start+test_length do
      printf("%s:%s\n",dict.index_to_symbol[x[i]], y[i])
    end
  else
    for i=start+1, start+test_length do
      printf("%s,",dict.index_to_symbol[x[i]])
    end
    printf("\n")
    for i=1,test_length do
      printf("%s,",y[i])
    end
    printf("\n")
  end
end

-- test_x_y(x,y,dict,30)
-- test_x_y(x,y,dict,40,true,x:size(1)-40)

model_name = 'Bidirectional LSTM'
model_class = Model
model_structure = 'bilstm'
nlayers = 1
mem_dim = 120
window_size = 5

header(model_name .. ' for Sentiment Classification')

model = model_class{
  dict = dict,
  structure = model_structure,
  num_layers = nlayers,
  mem_dim = mem_dim,
  window_size = window_size,
}

-- number of epochs to train
num_epochs = 1

header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

train_start = sys.clock()
best_dev_score = -1.0
best_dev_model = model
header('Training model')
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  model:train({x=x, y=y})
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)

  -- uncomment to compute train scores
  local train_predictions = model:predict_dataset(train_dataset)
  local train_score = accuracy(train_predictions, train_dataset.labels)
  printf('-- train score: %.4f\n', train_score)

  local dev_predictions = model:predict_dataset(dev_dataset)
  local dev_score = accuracy(dev_predictions, dev_dataset.labels)
  printf('-- dev score: %.4f\n', dev_score)

  if dev_score > best_dev_score then
    best_dev_score = dev_score
    best_dev_model = model_class{
      emb_vecs = vecs,
      structure = model_structure,
      fine_grained = fine_grained,
      num_layers = args.layers,
      mem_dim = args.dim,
    }
    best_dev_model.params:copy(model.params)
    best_dev_model.emb.weight:copy(model.emb.weight)
  end
end
printf('finished training in %.2fs\n', sys.clock() - train_start)
