--[[
  This is the CNN part for the method of CVPR paper MDNet: A Semantically and Visually Interpretable Medical Image Diagnosis Network.
  
  Zizhao Zhang @ U. of Florida
  zizhao@cise.ufl.edu

  Acknowledgement: The code is built upon the implementation of Wide Residual Network (https://github.com/szagoruyko/wide-residual-networks) by Sergey Zagoruyko
--]]

local nn = require 'nn'
local utils = paths.dofile'utils.lua'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function Dropout()
   return nn[opt.dropout_type](opt.dropout and opt.dropout or 0,nil,true)
end

local function createModel(opt)
   local depth = opt.depth
   
   -- The new Residual Unit in [a]
   local function bottleneck(nInputPlane, nOutputPlane, stride)
      
      local nBottleneckPlane = nOutputPlane / 4
      if opt.resnet_nobottleneck then
         nBottleneckPlane = nOutputPlane
      end

      -- BottleNeck residual block
      local convs = nn.Sequential()
      -- conv1x1
      convs:add(SBatchNorm(nInputPlane))
      convs:add(ReLU(true))
      convs:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
    
      -- conv3x3
      convs:add(SBatchNorm(nBottleneckPlane))
      convs:add(ReLU(true))
      convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,1,1,1,1))
    
      -- conv1x1
      convs:add(SBatchNorm(nBottleneckPlane))
      convs:add(ReLU(true))
      convs:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,1,1,0,0))

      
      if nInputPlane == nOutputPlane and stride == 1 then
        -- use identity mapping 
        local shortcut = nn.Identity()
        
        return nn.Sequential()
            :add(nn.ConcatTable()
               :add(convs)
               :add(shortcut))
            :add(nn.CAddTable(true))
      else
        -- Use ensemble-connection when feature map changes (when stride != 1 or nInputPlane != nOutputPlane)
        local shortcut = nn.Sequential()
        shortcut:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0))
         
        return nn.Sequential()
            :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
            :add(nn.JoinTable(2,4))
      end
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride)
      local down = down or false
      local s = nn.Sequential()
 
      s:add(block(nInputPlane, nOutputPlane, stride))
      nOutputPlane = nOutputPlane * 2
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   do
      assert((depth - 2) % 9 == 0, 'depth should be 9n+2')
      local n = (depth - 2) / 9

      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(layer(bottleneck, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(bottleneck, nStages[2]*2, nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(bottleneck, nStages[3]*2, nStages[4]/2, n, 2)) -- Stage 3 (spatial size: 8x8), the output feature map is divided by 2 to avoid parameter explosion
      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], opt.num_classes))
   end

   utils.DisableBias(model)
   utils.testModel(model)
   utils.MSRinit(model)
   utils.FCinit(model)

   return model
end

return createModel(opt)
