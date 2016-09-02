require 'nn'
require 'image'
require 'xlua'
require 'torch'
require 'cudnn'
require 'cunn'


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options:')
cmd:option('-videoHeight',240 ,'Height of the video frames')
cmd:option('-videoWidth',320 ,'Width of the video frames')
cmd:option('-batchSize', 50,'Frames per second')
cmd:text()

local opt = cmd:parse(arg)

local trsize = 39250 -- 11568
local tesize = 17650  -- 5272

local b = 1

for set = 1, 10 do

    print ("loading training Data .....")
    local trainData = torch.load('./train/trainData/trainData_' .. set .. '.t7')

    print ("loading testing Data .....")
    local testData = torch.load("./test/testData/testData_" .. set .. ".t7")

    trainData.data = trainData.data:double()
    testData.data = testData.data:double()

    print ("normalizing ....")

    local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))

    for i = 1,trainData.data:size()[1] do
        xlua.progress(i, trainData.data:size()[1])
        -- rgb -> yuv
        local rgb = trainData.data[i]
        local yuv = image.rgb2yuv(rgb)
        -- normalize y locally:
        yuv[1] = normalization(yuv[{{1}}])
        trainData.data[i] = yuv
    end

    -- normalize u globally:
    local mean_u = trainData.data:select(2,2):mean()
    local std_u = trainData.data:select(2,2):std()
    trainData.data:select(2,2):add(-mean_u)
    trainData.data:select(2,2):div(std_u)
    -- normalize v globally:
    local mean_v = trainData.data:select(2,3):mean()
    local std_v = trainData.data:select(2,3):std()
    trainData.data:select(2,3):add(-mean_v)
    trainData.data:select(2,3):div(std_v)

    trainData.mean_u = mean_u
    trainData.std_u = std_u
    trainData.mean_v = mean_v
    trainData.std_v = std_v

  -- preprocess testSet
    for i = 1,testData.data:size()[1] do
        xlua.progress(i, testData.data:size()[1])
        -- rgb -> yuv
        local rgb = testData.data[i]
        local yuv = image.rgb2yuv(rgb)
        -- normalize y locally:
        yuv[{1}] = normalization(yuv[{{1}}])
        testData.data[i] = yuv 
    end 
    -- normalize u globally:
    testData.data:select(2,2):add(-mean_u)
    testData.data:select(2,2):div(std_u)
    -- normalize v globally:
    testData.data:select(2,3):add(-mean_v)
    testData.data:select(2,3):div(std_v)

    
    trainData.data =  torch.reshape(trainData.data, trsize, 1, 3, 128, 128) 
    testData.data =  torch.reshape(testData.data, tesize, 1, 3, 128, 128) 


    --print ("shuffling ....")

    --[[
    local shuffle_tr = torch.randperm(trainData.data:size()[1])
    for i =1 , trainData.data:size()[1] do
        trainData.data[i] = trainData.data[shuffle_tr[i]]
      --  trainData.labels[i] = trainData.labels[shuffle_tr[i]]
    --end    
    
    --local shuffle_te = torch.randperm(testData.data:size()[1]) 
    --for i =1 , testData.data:size()[1] do
      --  testData.data[i] = testData.data[shuffle_te[i]]
    --    testData.labels[i] = testData.labels[shuffle_te[i]]
    --end      


print ('pairing up  training data.....')
print (set)

for b_train = (set-1)*40000/opt.batchSize + 1, (set)*40000/opt.batchSize do

    local catTrainData = {
        data = torch.CudaTensor(opt.batchSize, 2, 3, 128 , 128):fill(0),
        labels = torch.CudaTensor (opt.batchSize):fill(2),
        size = function() return opt.batchSize end
    }

    local index = 1

    for i = 1, opt.batchSize, 2 do
        --local paired_data_tr[index][1] = trainData.data[i]:clone()
        rand_index_f = math.random (1, trsize)
        rand_index =  math.random(1, trsize)
        for j = rand_index, trsize do
            if trainData.labels[rand_index_f] == trainData.labels[j] then

                catTrainData.data[index] = torch.cat( trainData.data[rand_index_f], trainData.data[j], 2)
                catTrainData.labels[index] = 1
                break
            end
        end
        index = index + 1
        rand_index_f = math.random (1, trsize)
        rand_index =  math.random(1, trsize)
        for j = rand_index , trsize do
            if trainData.labels[rand_index_f] ~= trainData.labels[j] then

                catTrainData.data[index] = torch.cat( trainData.data[rand_index_f], trainData.data[j], 2)
                catTrainData.labels[index] = 2
                break
            end
        end
        index = index + 1
        --print (index)
    end
    catTrainData.data = catTrainData.data:cuda()
    catTrainData.labels = catTrainData.labels:cuda()
    
    torch.save('./washDataset_10xAug/train/' .. b_train .. '.t7', catTrainData)
    
end

print ('pairing up  testing data.....')
print (set)

for b_test = (set-1)*4000/opt.batchSize + 1 , (set)*4000/opt.batchSize do
        
    local catTestData = {
        data = torch.CudaTensor(opt.batchSize, 2, 3, 128 , 128):fill(0),
        labels = torch.CudaTensor (opt.batchSize):fill(2),
        size = function() return opt.batchSize end
    }
    index = 1

    for i = 1, opt.batchSize, 2 do
        rand_index_f = math.random (1, tesize)
        rand_index =  math.random(1, tesize)

        for j = rand_index, tesize do
            if testData.labels[rand_index_f] == testData.labels[j] then
                catTestData.data[index] = torch.cat(testData.data[rand_index_f], testData.data[j], 2)
                catTestData.labels[index] = 1
                break
            end
        end
        index = index + 1

        rand_index_f = math.random (1, tesize)
        rand_index =  math.random(1, tesize)
        for j = rand_index, tesize do
            if testData.labels[rand_index_f] ~= testData.labels[j] then
                catTestData.data[index] = torch.cat(testData.data[rand_index_f], testData.data[j], 2)
                catTestData.labels[index] = 2
                break
            end
        end
        index = index + 1
    end
    catTestData.data = catTestData.data:cuda()
    catTestData.labels = catTestData.labels:cuda()

    torch.save('./washDataset_10xAug/test/' .. b_test .. '.t7', catTestData) 
end

end

