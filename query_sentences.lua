require 'torch'
require 'nngraph'
require 'nn'
require 'optim'
require 'base'

stringx = require('pl.stringx')
require 'io'

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  --for i = 2,#line do
   -- if line[i] ~= 'foo' then error({code="vocab", word = line[i]}) end
 -- end
  return line
end

function reset_state(state)
	layers = 2
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * layers do
            model.start_s[d]:zero()
        end
    end
end

function map_line(line)
	mapped_line={}
	for i =2,#line do
	mapped_line[i-1]=vocab_map[line[i]]
	end
	return torch.DoubleTensor(mapped_line)
end

function print_line(line)
for i=1,line:size(1) do
    io.write(inv_vocab_map[line[{i}]]," ") end
end

function run_test(line)
    reset_state(line)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = line.data:size(1)
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = torch.DoubleTensor(20):fill((line.data[i]))
        local y = torch.DoubleTensor(20):fill((line.data[i + 1]))
        perp_tmp, model.s[1],preds = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
	-- to avoid the unking issue
	xfer = nn.SoftMax()
	new_preds = xfer:forward(preds)
        ind = torch.multinomial(new_preds,1)
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
	-- get first prediction
	return ind[{1,1}]	
end

inv_vocab_map = torch.load('data/inv_vocab_map.tb') 
vocab_map = torch.load('data/vocab_map.tb') 
while true do
  print("Query: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      print("Word not in vocabulary:  ", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
	mapped = map_line(line)
line_ds = {}
line_ds.data = mapped
    for i = 1, line[1] do 
model_file = 'model_test_2.net'
model_dir = '.'
model = torch.load(paths.concat(model_dir,model_file))
next_preds = run_test(line_ds)
pred_word = inv_vocab_map[next_preds]
line_ds.data = line_ds.data:cat(torch.DoubleTensor({next_preds}))
end
print_line(line_ds.data)
    io.write('\n')
  end
end
