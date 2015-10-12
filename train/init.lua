require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

include('LSTM.lua')
include('model.lua')

printf = utils.printf

function share_params(cell, src)
  if torch.type(cell) == 'nn.gMoudle' then
    for i=1, #cell.forwardnodes do
      local node=cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gadBias')
  else
    error('parameters cannot be shared for ' .. torch.type(cell))
  end
end

function header(s)
  print(string.rep('-',80))
  print(s)
  print(string.rep('-',80))
end
