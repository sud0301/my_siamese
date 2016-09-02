--require 'rnn'
require 'optim'
--display = require 'display'

local UCFLoader = require 'UCFLoader'


-- parameters
--torch.manualSeed(1337)

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options:')

cmd:option('-learningRate', 0.0001,'Learning rate')
cmd:option('-beta1', 0.9,'First order momentum for adam')
cmd:option('-gradClip', 1.0,'Gradient clipped to')

cmd:option('-batchSize',1,'Batches trained in parallel, only 1 supported right now')

cmd:option('-printInterval',50,'Print every printInterval iterations')
cmd:option('-valInterval', 500,'Validate every valInterval iterations')


cmd:option('-mode','image','image....')

cmd:option('-gpuId',0,'Which GPU to use (-1 is CPU)')
cmd:option('-iterations', 4000*50,'Number of iterations')

cmd:option('-saveFile','net_wash10_1808.t7','Location for logging') -- to load: p = torch.load('test.t7')

cmd:option('-model', 'vgg_bn_drop_v3', 'model name')
cmd:text()

local opt = cmd:parse(arg)
local optimParams = {learningRate = opt.learningRate, beta1 = opt.beta1} 
local inputSize = 2048


local useCuda = (opt.gpuId >= 0)

if opt.gpuId >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpuId + 1)
end


do -- data augmentation module
    local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

    function BatchFlip:__init()
        parent.__init(self)
        self.train = true
    end

    function BatchFlip:updateOutput(input)
        if self.train then
            local bs = input:size(1)
            local flip_mask = torch.randperm(bs):le(bs/2)
            for i=1,input:size(1) do
                if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
            end
        end
        --print (input:size())
        self.output:set(input)
        return self.output
    end
end

-- load data and build model
local dataLoader = UCFLoader.new(opt, useCuda, false)
local model
local criterion

if opt.mode == 'image' then
    model = nn.Sequential()
    model:add(dofile('models/'..opt.model..'.lua'))
    
    criterion = nn.HingeEmbeddingCriterion(1) 
    --criterion = nn.CrossEntropyCriterion() --CrossEntropyCriterion())
end


if useCuda then
  model:cuda()
  criterion:cuda()
end

x, dldx = model:getParameters()

print(model)

function feval(x_new)
    if x ~= x_new then
        x:copy(x_new)
    end

    dldx:zero()
    local batch, targets = dataLoader:nextBatch(opt.batchSize)
    local output = model:forward(batch)
    local loss = criterion:forward(output, targets)
    local dfdo = criterion:backward(output, targets)
    model:backward(batch, dfdo)
  --dldx:clamp(-opt.gradClip, opt.gradClip)
  
  return loss, dldx
end

function validate()
  print('Validating...')
  local correctClassified = 0.0
  for k=1, dataLoader.numSequencesVal do 
    local batch, targets = dataLoader:getValidation(k)
    local output = model:forward(batch)
    print (output)
    local _, m = torch.max(output, 2)
    for i = 1, m:size(1) do
        if m[i][1] == targets[i] then
            correctClassified = correctClassified + 1.0
        end
    end
  end 
  return (correctClassified / (dataLoader.numSequencesVal*50))
end

local checkpoint = {}
checkpoint.i = {}
checkpoint.trainLosses = {}
checkpoint.valLosses = {}
local timer = torch.Timer()

for i = 1, opt.iterations do
    _, fs = optim.adam(feval, x, optimParams)
  
        --model:zeroGradParameters()
        --model:forget()

    if i % opt.printInterval == 0 or i == opt.iterations or i==1 then
        local trainError = fs[1] 
        print('Iteration ' .. i) 
        print('Progress: ' .. string.format("%.2f", 100*i/opt.iterations) .. '%. Time for last interval: ' .. string.format("%.2f", timer:time().real) .. 's')
        timer:reset()
        print('Training error: ' .. trainError)

    -- just save loses for now
        table.insert(checkpoint.i, i)
        table.insert(checkpoint.trainLosses, trainError)

        torch.save(opt.saveFile, checkpoint)
    end

    if i % opt.valInterval == 0 then

        valAccuracy = validate()
        --model:zeroGradParameters()
        --model:forget()

        print('Validation Accuracy: ' .. valAccuracy .. '\n')

        table.insert(checkpoint.valLosses, valAccuracy)
        checkpoint.net = x

        checkpoint.opt = opt

        torch.save(opt.saveFile, checkpoint)
    end
end

checkpoint.net = x

checkpoint.opt = opt
torch.save(opt.saveFile, checkpoint)
