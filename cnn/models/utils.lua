local utils = {}

function utils.MSRinit(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nInputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
end

function utils.FCinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.bias:zero()
   end
end

function utils.DisableBias(model)
   for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
      v.bias = nil
      v.gradBias = nil
   end
end

function utils.testModel(model)
   model:float()
   local imageSize = opt and opt.imageSize or 32
   local input = torch.randn(1,3,imageSize,imageSize):type(model._type)
   print('forward output',{model:forward(input)})
   print('backward output',{model:backward(input,model.output)})
   model:reset()
end

function utils.makeDataParallelTable(model, nGPU)
   if nGPU > 1 then
      -- local gpus = torch.range(1, nGPU):totable()
      local gpus = torch.LongTensor{1,3}:totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end
   return model
end
---------------------------------------------------------------
-- checkpoint
---------------------------------------------------------------
utils.checkpoint = {}

function utils.checkpoint.load(filepath)
   print(c.blue '==> Loading checkpoint ' .. filepath..'.t7')
   local latest = torch.load(filepath..'.t7')
   return latest --, optimState
end

function utils.checkpoint.save(epoch, model, optimState, isBestModel, opt)
   -- Remove temporary buffers to reduce checkpoint size
   model:clearState()

   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   local modelFile = 'model_' .. epoch .. '.t7'
   local optimFile = 'optimState_' .. epoch .. '.t7'

   torch.save(paths.concat(opt.save, modelFile), model)
   torch.save(paths.concat(opt.save, optimFile), optimState)
   torch.save(paths.concat(opt.save, 'latest.t7'), {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   })

   if isBestModel then
      torch.save(paths.concat(opt.save, 'model_best_'..epoch..'.t7'), model)
   end

   -- Re-use gradInput buffers if the option is set. This is necessary because
   -- of the model:clearState() call clears sharing.
   if opt.shareGradInput then
      local models = require 'models/init'
      models.shareGradInput(model)
   end
end

return utils
