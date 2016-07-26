require 'nn'

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

ConvBNReLU(3,64):add(nn.Dropout(0.3))
ConvBNReLU(64,64)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(64,128):add(nn.Dropout(0.4))
ConvBNReLU(128,128)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(128,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256)
--vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(256,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
--vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
--vgg:add(MaxPooling(2,2,2,2):ceil())
vgg:add(nn.View(512*8*8))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512*8*8, 8192))
classifier:add(nn.BatchNormalization(8192))
classifier:add(nn.ReLU(true))
--classifier:add(nn.Dropout(0.5))
--classifier:add(nn.Linear(512,128))
--classifier:add(nn.BatchNormalization(128))
--classifier:add(nn.ReLU(true))
--classifier:add(nn.Dropout(0.5))
--classifier:add(nn.Linear(512, 2))

vgg:add(classifier)

siamese_model= nn.ParallelTable()
siamese_model:add(vgg)
siamese_model:add(vgg:clone('weight', 'bias', 'gradWeight', 'gradBias'))

siamese_vgg = nn.Sequential()
siamese_vgg:add(nn.SplitTable(2))
siamese_vgg:add(siamese_model)

--siamese_vgg:add(nn.JoinTable(2))
siamese_vgg:add(nn.CSubTable())
siamese_vgg:add(nn.Linear(8192, 4096))
--siamese_vgg:add(nn.Linear(1024, 512))
siamese_vgg:add(nn.BatchNormalization(4096))
siamese_vgg:add(nn.ReLU(true))
siamese_vgg:add(nn.Dropout(0.5))
siamese_vgg:add(nn.Linear(4096,1024))
siamese_vgg:add(nn.BatchNormalization(1024))
siamese_vgg:add(nn.ReLU(true))
siamese_vgg:add(nn.Dropout(0.5))
siamese_vgg:add(nn.Linear(1024, 2)) 

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(siamese_vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return siamese_vgg
