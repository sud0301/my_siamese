--require 'ffmpeg'
require 'torch'
require 'image'
require 'paths'
require 'cudnn'
require 'cunn'
--local t = require 'datasets/transforms'

local UCFLoader = {}
UCFLoader.__index = UCFLoader

function UCFLoader.new(opt, useCuda, creatureFlowFiles)

    local self = setmetatable({}, UCFLoader)

    self.opt = opt
    self.useCuda = useCuda

    self.currentIndex = 1
    self.numSequences = 8000 
    self.randomPerm = torch.randperm(self.numSequences)
    self.numSequencesVal = 800
    return self
end

function UCFLoader:nextBatch(numberSeq)
   if (self.currentIndex + numberSeq - 1) > self.numSequences then
        self.currentIndex = 1
        self.randomPerm = torch.randperm(self.numSequences)
    end
    if self.opt.mode == 'image' then

        local file = torch.load('washDataset_10xAug/train/' .. self.randomPerm[self.currentIndex] .. '.t7')
        local inputs = file.data
        local targets = file.labels
        self.currentIndex = self.currentIndex + numberSeq
        return inputs, targets
    end
end

function UCFLoader:getValidation(i)

    if self.opt.mode == 'image' then

        local file = torch.load('washDataset_10xAug/test/' .. i .. '.t7')
        local inputs = file.data
        local targets = file.labels

        return inputs, targets
    end
end



return UCFLoader
