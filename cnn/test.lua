-- require 'cutorch'
-- cutorch.setDevice(1)
require 'xlua'
require 'optim'
require 'image'
--require 'cunn'
 -- require 'cudnn'
require 'nngraph'
torchtools = require 'torchtools'
local optnet = require 'optnet'
graphgen = require 'optnet.graphgen'
iterm = require 'iterm'
require 'iterm.dot'
util = dofile('torchtools.lua')

opt = {
  dataset = './datasets/cifar10_whitened.t7',
  save = 'logs',
  batchSize = 128,
  learningRate = 0.1,
  learningRateDecay = 0,
  learningRateDecayRatio = 0.2,
  weightDecay = 0.0005,
  dampening = 0,
  momentum = 0.9,
  epoch_step = "80",
  max_epoch = 300,
  model = '',
  optimMethod = 'sgd',
  init_value = 10,
  shortcutType = 'A',
  nesterov = false,
  dropout = 0,
  hflip = true,
  randomcrop = 4,
  imageSize = 32,
  randomcrop_type = 'zero',
  cudnn_fastest = true,
  cudnn_deterministic = false,
  optnet_optimize = true,
  generate_graph = false,
  multiply_input_factor = 1,
  widen_factor = 4,
  depth = 110,
  nGPU = 1,
  num_output = 4
}
opt.num_classes = 10

net = dofile('models/ecnet.lua')

-- net = torch.load('logs/arxiv/cifar10/nin_graph_1129629721/model.t7')
-- net = torch.load('logs/nin_graph_wskeeper_small_2857512732_original/model.t7')

-- sample_input = torch.rand(1,3,32,32):float()
-- -- iterm.dot(graphgen(net, sample_input), 'tmp_graph.pdf')
-- -- -- 

-- out = net:forward(sample_input)
-- out = net:backward(sample_input, net.output)

-- -- torchtools.printBlobs(net)
-- -- torchtools.printNet(net,1)
local parameters,gradParameters = net:getParameters()
print('Network has ', parameters:numel(), 'parameters')
print('Network has', #net:findModules'nn.SpatialConvolution', 'convolutions')
