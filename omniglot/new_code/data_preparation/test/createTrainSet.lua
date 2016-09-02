--require 'ffmpeg'
require 'torch'
require 'image'
require 'paths'
--require 'cudnn'
require 'cunn'

local function split(inputstr, sep)
    if sep == nil then
            sep = "%s"
    end
    local t={} ; local i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
            t[i] = str
            i = i + 1
    end
    return t
end

function createClassIdx()
    local classIdx = {}
    local file, err = io.open('classIndE.txt', 'rb')
    if err then
            print('error opening the file')
            return
    else
            while true do
                local line = file:read()
                if line == nil then
                        break
                end
                -- get tokens from line containing video path and label
                local tokens = {}
                for token in string.gmatch(line, "([^%s]+)") do
                        table.insert(tokens, token)
                        
                end
                local idx, cl = unpack(tokens)
                    classIdx[cl] = tonumber(idx)
            end
    end
    return classIdx
end

function loadList(imgListPath)
    local classIdx = createClassIdx()   
    --print (classIdx)
    local imgPaths = {}
    local imgLabels = {}
    local file, err = io.open(imgListPath, 'rb')
    if err then
            print('error opening the file')
            return
    else
            while true do
                local line = file:read()
                if line == nil then
                    break
                end
                -- get tokens from line containing video path and label
                local tokens = {}
                for token in string.gmatch(line,'([^/]+)') do
                    --print (token)    
                    table.insert(imgLabels, classIdx[token])
                    break
                end
                table.insert(imgPaths, line)
            end
    end
     
    return imgPaths, imgLabels
end

imgPaths, imgLabels =  loadList('testing_paths.txt')

max_index_tr = #imgLabels
    if max_index_tr % 100 ~=0 then
        max_index_tr = max_index_tr - (max_index_tr % 100)
    end 

print ("creating tensor ... ")

local trainData = { 
        data = torch.Tensor(max_index_tr, 1, 128, 128),
        labels = torch.Tensor(max_index_tr),
        size = function() return max_index_tr end 
    }   

for i=1, max_index_tr do
    imgPaths[i] =  imgPaths[i]:gsub("character", "/character")
    trainData.data[i] = image.load(imgPaths[i], 1, 'float') 
    trainData.labels[i] = imgLabels[i]
    print (i)
end

print ("shuffling ...")

local shuffle_tr = torch.randperm(max_index_tr)
for i =1 , max_index_tr do
    trainData.data[i] = trainData.data[shuffle_tr[i]]
    trainData.labels[i] = trainData.labels[shuffle_tr[i]]
end   

 
print ("creating batches ...")

for b = 1, 2 do

    local batchData = { 
        data = torch.Tensor(max_index_tr/2, 1, 128, 128),
        labels = torch.Tensor(max_index_tr/2),
        size = function() return max_index_tr/2 end 
    }   
    local count = 1 

    for i= (b-1)*(max_index_tr/2)+1, b*max_index_tr/2 do
    
        batchData.data[count] = trainData.data[i]
        batchData.labels[count] = trainData.labels[i]
        count = count + 1 
    end 
    
    torch.save('testData_' .. b .. '.t7', batchData)   
end

--[[
local trainData = {
        data = torch.Tensor(#imgLabels, 1, 128, 128),
        labels = torch.Tensor(#imgLabels),
        size = function() return #imgLabels end
    }

for i=1, #imgLabels do
    imgPaths[i] =  imgPaths[i]:gsub("character", "/character")
    trainData.data[i] = image.load(imgPaths[i], 1, 'float') 
    trainData.labels[i] = imgLabels[i]
end

torch.save('trainData.t7', trainData)   
]]--
--[[
imgPathsVal, imgLabelsVal =  loadList('testing_paths.txt')

local testData = {
        data = torch.Tensor(#imgLabelsVal, 3, 105, 105),
        labels = torch.Tensor(#imgLabelsVal),
        size = function() return #imgLabelsVal end
    }

for i=1, #imgLabelsVal do 
    testData.data[i] = image.load(imgPathsVal[i], 3, 'byte') 
    testData.labels[i] = imgLabelsVal[i]
end

torch.save('testData_p1.t7', testData)   
--]]
