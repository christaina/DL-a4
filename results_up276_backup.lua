require 'optim'
require 'os'
gpu = false
if gpu then
    require 'cunn'
    print("Running on GPU") 
    
else
    require 'nn'
    print("Running on CPU")
end

require('nngraph')
require('base')
ptb = require('data')

datestamp = os.date()
timestamp = os.time()
logdir = 'logs_GRU'
testLogger = optim.Logger(paths.concat(logdir, timestamp..'_finaltest.log'))
testLogger:add{"final_test  :  "..datestamp}


model = torch.load("4_gru/model.net")

local params = {
                batch_size=20, -- minibatch
                seq_length=20, -- unroll length
                layers=2,
                decay=2,
                rnn_size=200, -- hidden unit size
                dropout=0, 
                init_weight=0.1, -- random weight initialization limits
                lr=1, --learning rate
                vocab_size=10000, -- limit on the vocabulary size
                max_epoch=4,  -- when to start decaying learning rate
                max_max_epoch=13, -- final epoch
                max_grad_norm=5 -- clip when gradients exceed this norm value
               }

function transfer_data(x)
    if gpu then
        return x:cuda()
    else
        return x
    end
end


function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 do
            model.start_s[d]:zero()
        end
    end
end


function run_test()
    print("Model is being tested on Test data...It may take few minutes...Grab a cup of coffee till model gives you the results...")
    testLogger:add{"Model is being tested on Test data...It may take few minutes...Grab a cup of coffee till model gives you the results..."}

    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    print("Test data size : ",len)
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1],preds = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    testLogger:add{"Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1)))}
    g_enable_dropout(model.rnns)
end

if gpu then
    g_init_gpu(arg)
end

state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
run_test()

