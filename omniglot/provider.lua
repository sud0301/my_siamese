require 'nn'
require 'image'
require 'xlua'
require 'torch'

--local Provider = torch.class 'Provider'

--function Provider:__init(full)
    local trsize = 11568
    local tesize = 5272

    -- load dataset
    local trainData = {
        data = torch.Tensor(11568, 1, 105, 105),
        labels = torch.Tensor(11568),
        size = function() return trsize end
    }
    --local trainData = self.trainData    
    --trainData= torch.load('data/mnist.t7/train_32x32.t7', 'ascii')
    trainData = torch.load('trainData.t7')

    local testData = {
        data = torch.Tensor(5272, 1, 105, 105),
        labels = torch.Tensor(5272),
        size = function() return tesize end
    }
    --local testData = self.testData
    --testData = torch.load('data/mnist.t7/test_32x32.t7', 'ascii')
    testData = torch.load('testData.t7')
    
    -- resize dataset (if using small version)
    trainData.data = trainData.data[{ {1,trsize} }]
    trainData.labels = trainData.labels[{ {1,trsize} }]

    testData.data = testData.data[{ {1,tesize} }]
    testData.labels = testData.labels[{ {1,tesize} }]

    -- reshape data
    trainData.data = trainData.data:reshape(trsize, 1, 105, 105)
    testData.data = testData.data:reshape(tesize, 1, 105, 105)
--end

--function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  -- 
    --local trainData = self.trainData
    --local testData = self.testData 
    
    --print (TrainData.data[1])  
    
    print '<trainer> preprocessing data (color space + normalization)'
    collectgarbage()

    -- preprocess trainSet
    --[[local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
    
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
    testData.data:select(2,3):div(std_v) ]]--
    
    --[[local mean_x = trainData.data:mean()
    local std_x = trainData.data:std()
    
    trainData.data:add(-mean_x)
    trainData.data:div(std_x)
    
    testData.data:add(-mean_x)
    testData.data:div(std_x)
    ]]--
    trainData.data =  torch.reshape(trainData.data ,11568, 1, 1, 105, 105) 
    testData.data =  torch.reshape(testData.data, 5272, 1, 1, 105, 105) 

    shuffle_tr = torch.randperm(trainData.data:size()[1])
    for i =1 , trainData.data:size()[1] do
        trainData.data[i] = trainData.data[shuffle_tr[i]]
        trainData.labels[i] = trainData.labels[shuffle_tr[i]]
    end    
    
    shuffle_te = torch.randperm(testData.data:size()[1]) 
    for i =1 , testData.data:size()[1] do
        testData.data[i] = testData.data[shuffle_te[i]]
        testData.labels[i] = testData.labels[shuffle_te[i]]
    end      

    --print(trainData.data[1])
    --print (trainData.labels)
    
    local catTrainData = {
        data = torch.Tensor(30000, 2, 1, 105 , 105):fill(0),
        labels = torch.Tensor (30000):fill(2),
        size = function() return 30000 end
    }
    --local catTrainData = self.catTrainData
     
    local catTestData = {
        data = torch.Tensor(10000, 2, 1, 105, 105):fill(0),
        labels = torch.Tensor (10000):fill(2),
        size = function() return 10000 end
    } 
    --local catTestData = self.catTestData 
    
    max_index_tr = catTrainData.data:size()[1]
    if max_index_tr % 2 ~=0 then
        max_index_tr = max_index_tr - 1
    end

    max_index_te = catTestData.data:size()[1]
    if max_index_te % 2 ~=0 then
        max_index_te = max_index_te - 1
    end

    local index = 1

    print ('pairing up .....')   
 
    for i = 1, max_index_tr, 2 do
        --local paired_data_tr[index][1] = trainData.data[i]:clone()
        rand_index_f = math.random (1, trsize) 
        rand_index =  math.random(1, trsize)
        for j = rand_index, trsize do
            if trainData.labels[rand_index_f] == trainData.labels[j] then
                catTrainData.data[index] = torch.cat(trainData.data[rand_index_f], trainData.data[j], 2)
                catTrainData.labels[index] = 1
                --print (catTrainData.labels[index])
        --        print ('111111111')
                break
            end
        end
        index = index + 1
        
        rand_index_f = math.random (1, trsize) 
        rand_index =  math.random(1, trsize)
        for j = rand_index, trsize do
            if trainData.labels[rand_index_f] ~= trainData.labels[j] then
                catTrainData.data[index] = torch.cat(trainData.data[rand_index_f], trainData.data[j], 2)
                catTrainData.labels[index] = 2
                --print (catTrainData.labels[index])
          --      print ('222222222')
                break
            end
        end
        index = index + 1
    end
   
    --print ('input: ') 
    --answer=io.read()
     
    index = 1
    for i = 1, max_index_te, 2 do
        rand_index_f = math.random (1, tesize) 
        rand_index =  math.random(1, tesize)
        --print ('rand_index_f: ' .. rand_index_f)
        --print ('rand_index: ' .. rand_index)
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
    print ('saving the tensor')
     
    torch.save("trainSet.t7", catTrainData)
    torch.save("testSet.t7", catTestData)
--end
