require('init')

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

model_name = 'Bidirectional LSTM'
model_class = LSTM


