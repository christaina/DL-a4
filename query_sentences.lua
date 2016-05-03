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
    --state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1,  layers do
            model.start_s[d]:zero()
        end
    end
end

function map_line(line)
	mapped_line={}
	for i =2,#line do
	mapped_line[i-1]=vocab_map[line[i]]
	end
	--return torch.DoubleTensor(mapped_line)
	return mapped_line
end

function print_line(line)
for i=1,line:size(1) do
    io.write(inv_vocab_map[line[{i}]]," ") end
end

function run_test(line)
--next_word = ""
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = #line
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
        local x = line
        local y = line
        perp_tmp, model.s[1],preds = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
	-- to avoid the unking issue
	xfer = nn.SoftMax()
	new_preds = xfer:forward(preds)
        local ind = torch.multinomial(new_preds,1)
        perp = perp + perp_tmp[1] 
        g_replace_table(model.s[0], model.s[1])
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
model_file = 'model.net'
model_dir = '4_gru'
model = torch.load(paths.concat(model_dir,model_file))
line_ds = {}
line_ds.data = mapped
reset_state()
for i = 1, #mapped do
io.write(inv_vocab_map[mapped[i]]," ")
curr = torch.DoubleTensor(20):fill(mapped[i])
pred = run_test(curr)
next_out = inv_vocab_map[pred]
end
    for i = 1, line[1] do 
curr = (torch.DoubleTensor(20):fill(pred))
pred = run_test(curr)
next_out= inv_vocab_map[pred]
io.write(next_out," ")
end
    io.write('\n')
  end
end
