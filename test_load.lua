require 'torch'
require 'nn'
require 'nngraph'

m_Name = 'model_test_2.net'
m_orig = 'model.net'
mod = torch.load(m_orig)
print(mod.core_network)
