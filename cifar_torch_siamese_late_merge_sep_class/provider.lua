require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class 'Provider'

function Provider:__init(full)
    local trsize = 50000
    local tesize = 10000

    -- download dataset
    if not paths.dirp('cifar-10-batches-t7') then
        local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
        local tar = paths.basename(www)
        os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
    end 

    -- load dataset
    self.trainData = { 
        data = torch.Tensor(50000, 3072),
        labels = torch.Tensor(50000),
        size = function() return trsize end 
    }   
    local trainData = self.trainData
    for i = 0,4 do
        local subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
        trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
        trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    end 
    trainData.labels = trainData.labels + 1 

    local subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
    self.testData = { 
        data = subset.data:t():double(),
        labels = subset.labels[1]:double(),
        size = function() return tesize end 
    }   
    local testData = self.testData
    testData.labels = testData.labels + 1 

    -- resize dataset (if using small version)
    trainData.data = trainData.data[{ {1,trsize} }]
    trainData.labels = trainData.labels[{ {1,trsize} }]

    testData.data = testData.data[{ {1,tesize} }]
    testData.labels = testData.labels[{ {1,tesize} }]

    -- reshape data
    trainData.data = trainData.data:reshape(trsize,3,32,32)
    testData.data = testData.data:reshape(tesize,3,32,32)

    print (trainData.data[1])

    trsize = 30000
    tesize = 20000
    
    self.newTrainData = {
        data = torch.Tensor(30000, 3, 32 ,32),
        labels = torch.Tensor(30000),
        size = function() return trsize end
    } 
    local newTrainData = self.newTrainData    
      
    self.newTestData = {
        data = torch.Tensor(20000, 3, 32 ,32),
        labels = torch.Tensor(20000),
        size = function() return tesize end
    } 
    local newTestData = self.newTestData
        
      
    local index_tr = 1
    local index_te = 1
    for i = 1 , trainData.data:size()[1] do
        if trainData.labels[i] == 7 or trainData.labels[i] == 8 or trainData.labels[i] == 9 or trainData.labels[i] == 10 then
            if index_te == 20001 then
                break
            end
            newTestData.data[index_te] = trainData.data[i]
            newTestData.labels[index_te] = trainData.labels[i]
            index_te = index_te + 1
        else
            if index_tr == 30001 then
                break
            end
            newTrainData.data[index_tr] = trainData.data[i]
            newTrainData.labels[index_tr] = trainData.labels[i]
            index_tr = index_tr + 1
            
        end
    end
end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
    local newTrainData = self.newTrainData
    local newTestData = self.newTestData

    print '<trainer> preprocessing data (color space + normalization)'
    collectgarbage()

    --[[ preprocess trainSet
    local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
    for i = 1,trainData:size() do
        xlua.progress(i, trainData:size())
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
    for i = 1,testData:size() do
        xlua.progress(i, testData:size())
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
    ]]--
    local mean_x = newTrainData.data:mean()
    local std_x = newTrainData.data:std()
    
    newTrainData.data:add(-mean_x)
    newTrainData.data:div(std_x)
    
    newTestData.data:add(-mean_x)
    newTestData.data:div(std_x)
    
    newTrainData.data =  torch.reshape(newTrainData.data ,30000, 1, 3, 32, 32) 
    newTestData.data =  torch.reshape(newTestData.data, 20000, 1, 3, 32, 32) 

    self.catTrainData = {
        data = torch.Tensor(15000, 2, 3, 32 , 32),
        labels = torch.Tensor (15000),
        size = function() return trsize/2 end
    }
    local catTrainData = self.catTrainData
     
    self.catTestData = {
        data = torch.Tensor(10000, 2, 3, 32 , 32),
        labels = torch.Tensor (10000),
        size = function() return tesize/2 end
    }
    
    local catTestData = self.catTestData 
    
    max_index_tr = catTrainData.data:size()[1]
    if max_index_tr % 2 ~=0 then
        max_index_tr = max_index_tr - 1
    end

    max_index_te = catTestData.data:size()[1]
    if max_index_te % 2 ~=0 then
        max_index_te = max_index_te - 1
    end

    --[[paired_data_tr = torch.Tensor(max_index_tr/2, 2, trainData.data:size()[2], trainData.data:size()[3], trainData.data:size()[4])
    paired_labels_tr = torch.Tensor(max_index_tr/2)

    paired_data_te = torch.Tensor(max_index_te/2, 2, testData.data:size()[2], testData.data:size()[3], testData.data:size()[4])
    paired_labels_te = torch.Tensor(max_index_te/2)
    ]]--

    --[[print (paired_data_tr:size())
    print (paired_labels_tr:size())

    print (paired_data_te:size())
    print (paired_labels_te:size())]]--

    index = 1
    count_pos = 1
    count_neg = 1

    for i = 1, max_index_tr*2, 4 do
        --local paired_data_tr[index][1] = trainData.data[i]:clone()
        rand_index =  math.random(1, max_index_tr)
        for j = rand_index, max_index_tr do
            if newTrainData.labels[i] == newTrainData.labels[j] then
                catTrainData.data[index] = torch.cat(newTrainData.data[i], newTrainData.data[j], 2)
                catTrainData.labels[index] = 1
                break
            end
        end
        

        index = index + 1
        rand_index =  math.random(1, max_index_tr)
        for j = rand_index, max_index_tr do
            if newTrainData.labels[i+2] ~= newTrainData.labels[j] then
                catTrainData.data[index] = torch.cat(newTrainData.data[i+2], newTrainData.data[j], 2)
                catTrainData.labels[index] = 2
                break
            end
        end

        index = index + 1
    end

    --local dataset_train = {}
    --dataset_train.data = paired_data_tr
    --dataset_train.labels = paired_labels_tr

--print (dataset_train.data:size())
    local index = 1

    for i = 1, max_index_te*2, 4 do
        --paired_data_te[index][1] = testData.data[i]:clone()
        rand_index =  math.random(1, max_index_te)
        for j = rand_index, max_index_te do
            if newTestData.labels[i] == newTestData.labels[j] then
                --paired_data_te[index][2] = testData.data[j]:clone()
                --paired_labels_te[index] = 1
                catTestData.data[index] = torch.cat(newTestData.data[i], newTestData.data[j], 2)
                catTestData.labels[index] = 1
                break
            end
        end

        index = index + 1
        rand_index =  math.random(1, max_index_te)
        for j = rand_index, max_index_te do
            if newTestData.labels[i+2] ~= newTestData.labels[j] then
                catTestData.data[index] = torch.cat(newTestData.data[i+2], newTestData.data[j], 2)
                catTestData.labels[index] = 2
                break
            end
        end

        index = index + 1
    end
end
