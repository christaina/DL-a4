stringx = require('pl.stringx')
require 'io'
require('nngraph')
require('base')
gpu = false
if gpu then
    require 'cunn'
    print("Running on GPU") 
    
else
    require 'nn'
    print("Running on CPU")
end

function reset_state()
   num_layers = 2 
    if model ~= nil and model.start_s ~= nil then
        for d = 1, num_layers do
            model.start_s[d]:zero()
        end
    end
end

function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

function transfer_data(x)
    if gpu then
        return x:cuda()
    else
        return x
    end
end

function map_line(line)
	mapped_line={}
	for i =2,#line do
	mapped_line[i-1]=vocab_map[line[i]]
	end
        return mapped_line
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  if line == 'Q' then error({code="Q"}) end
  line = stringx.split(line)
  no_of_predictions = tonumber(line[1])
  if no_of_predictions == nil then error({code="init"}) end
  if line[2] == nil then error({code="At least one word required"}) end
  return line
end

function run_test(line)
	g_disable_dropout(model.rnns)
	local perp = 0
	g_replace_table(model.s[0], model.start_s)
        local x = line
        local y = torch.DoubleTensor(20):fill(1)
	perp_tmp, model.s[1],preds = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
	xfer = nn.SoftMax()
	new_preds = xfer:forward(preds)
	ind = torch.multinomial(new_preds,1)[{1,1}]
	perp = perp + perp_tmp[1]
	g_replace_table(model.s[0], model.s[1])
	return ind	
end
while true do
  print("\nQuery Format: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "Q" then
      break
    elseif line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      print("\nWord not in vocabulary: ", line.word)
    elseif line.code == "init" then
      print("\nStart with a number")
    else
      print(line)
      print("\nFailed, try again")
    end
  else
	model_name = 'model.net'
	model_dir = '4_gru'
	data_dir = 'data'
	inv_map = 'inv_vocab_map.tb'
	vocab_map='vocab_map.tb'
    model = torch.load(paths.concat(model_dir,model_name))
    inv_vocab_map = torch.load(paths.concat(data_dir,inv_map))
    vocab_map = torch.load(paths.concat(data_dir,vocab_map))
    mapped = map_line(line)
	reset_state()
	-- iterate through past
    for i = 2,#line do
	io.write(line[i]," ")
        x1 = mapped[i-1]
        input = torch.Tensor(20):fill(x1) 
        op = run_test(input)
        next_word = inv_vocab_map[op]
    end
	-- make preds
    for i = 1, line[1] do
        input = torch.Tensor(20):fill(op)
        op = run_test(input)
        next_word = inv_vocab_map[op]
        io.write(next_word," ")
    end
    io.write('\n')
  end
end
