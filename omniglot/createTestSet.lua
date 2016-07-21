--require 'ffmpeg'
require 'torch'
require 'image'
require 'paths'
require 'cudnn'
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
    local file, err = io.open('classInd.txt', 'rb')
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
                    table.insert(imgLabels, classIdx[token])
                    break
                end
                table.insert(imgPaths, line)
            end
    end
     
    return imgPaths, imgLabels
end

--[[
imgPaths, imgLabels =  loadList('training_paths.txt')

local trainData = {
        data = torch.Tensor(#imgLabels, 3, 105, 105),
        labels = torch.Tensor(#imgLabels),
        size = function() return #imgLabels end
    }

for i=1, #imgLabels do 
    trainData.data[i] = image.load(imgPaths[i], 3, 'byte') 
    trainData.labels[i] = imgLabels[i]
end

torch.save('trainData.t7', trainData)   
]]--
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

torch.save('testData.t7', testData)   
