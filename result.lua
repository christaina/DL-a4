require 'torch'
require 'nngraph'
require 'io'
require 'nn'

local stringx = require('pl.stringx')
local file = require('pl.file')

model = 'model_test_gru4.net'
modeldir = 'logs_gru_newlr_long'
datadir = 'data'
testdata = 'ptb.test.txt'
vocab_map= 'vocab_map.tb'

vocab_map = torch.load(paths.concat(datadir,vocab_map))

function reset_state(state)
        layers = 2
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
	-- make 2 x layers for LSTM implementation
        for d = 1,  layers do
            model.start_s[d]:zero()
        end
    end
end

function g_disable_dropout(node)
    if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(g_disable_dropout)
        end
        return
    end
    if string.match(node.__typename, "Dropout") then
        node.train = false
    end
end

function g_replace_table(to, from)
    assert(#to == #from)
    for i = 1, #to do
        to[i]:copy(from[i])
    end
end

function map_line(line)
        mapped_line={}
        for i =2,#line do
        mapped_line[i-1]=vocab_map[line[i]]
        end
        return torch.DoubleTensor(mapped_line)
end

function g_f3(f)
    return string.format("%.3f", f)
end

function run_test()
	print("starting...")
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
	local prog = len/100
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
--	if i % prog == 0 then
--		print("step" ) end
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1],preds = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
--print("predictions look like...")
--print(ind)
--print_preds(ind)
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
end


local function load_data(fname)
    local data = file.read(fname)
    data = stringx.replace(data, '\n', '<eos>')
    data = stringx.split(data)
    --print(string.format("Loading %s, size of data = %d", fname, #data))
    local x = torch.zeros(#data)
    for i = 1, #data do
        x[i] = vocab_map[data[i]]
    end
    return x
end


data = load_data(paths.concat(datadir,testdata))
model = torch.load(paths.concat(modeldir,model))
bs = 20
data = data:resize(data:size(1),1):expand(data:size(1),bs)
--leftovers = data:size(1)%bs

-- make data size multiple of 20
--ext = torch.DoubleTensor(bs-leftovers):fill(0)
--data = data:cat(ext)
state_test = {}
state_test.data = data
run_test()
-- need to divide data into batches of 20
-- do i need to get perplexity of each group of 20
-- need to load data into state var
