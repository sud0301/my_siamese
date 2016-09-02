require 'torch'
require 'image'
--require 'ffmpeg'


local UCFLoader = require 'UCFLoader'

cmd = torch.CmdLine()
cmd:text('Options:')

--cmd:option('-batchSize',5 ,'Maximum absolute optical flow value')
--cmd:option('-videoWidth',320 ,'Width of the video frames')
--cmd:option('-videoHeight',240 ,'Height of the video frames')
--cmd:option('-desiredFPS', 25,'Frames per second')

cmd:option('-gpuId',0,'Which GPU to use (-1 is CPU)')

local opt = cmd:parse(arg)

local useCuda = (opt.gpuId >= 0)

if opt.gpuId >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuId + 1)
end

local dataLoader = UCFLoader.new(opt, useCuda, false)

--x = dataLoader:nextBatch(5)
