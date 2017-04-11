--------------------------------------------------------------------------------
-- Subclass of BaseDataLoader that provides data from two datasets.
-- The samples from the datasets are not aligned.
-- The datasets can have different sizes
--------------------------------------------------------------------------------
require 'data.base_data_loader'
require 'image'


local class = require 'class'
data_util = paths.dofile('data_util.lua')

StdinUnalignedDataLoader = class('StdinUnalignedDataLoader', 'BaseDataLoader')

function StdinUnalignedDataLoader:__init(conf)
  BaseDataLoader.__init(self, conf)
  conf = conf or {}
end

function StdinUnalignedDataLoader:name()
  return 'StdinUnalignedDataLoader'
end

function StdinUnalignedDataLoader:Initialize(opt)
  opt.align_data = 0
  self.opt = opt
end

-- actually fetches the data
-- |return|: a table of two tables, each corresponding to
-- the batch for dataset A and dataset B
function StdinUnalignedDataLoader:LoadBatchForAllDatasets()
  print("Please enter path to image to convert:")
  local pathA = {io.read()}
  local batchA = {self.loadSingleImage(self, pathA[1], self.opt)}

  return batchA, batchA, pathA, pathA
end

-- returns the size of each dataset
function StdinUnalignedDataLoader:size(dataset)
  return 1
end

function StdinUnalignedDataLoader:loadSingleImage(path, opt)
    print("OPENING IMAGE", path)

    local input_nc = opt.nc -- input channels
    local loadSize   = {input_nc, opt.loadSize}
    local sampleSize = {input_nc, opt.fineSize}

    local im = image.load(path, opt.input_nc, 'float')
    if opt.resize_or_crop == 'resize_and_crop' then
      im = image.scale(im, loadSize[2], loadSize[2])
    end
    if input_nc == 3 then
      local perm = torch.LongTensor{3, 2, 1}
      im = im:index(1, perm)--:mul(256.0): brg, rgb
      im = im:mul(2):add(-1)
    end
    assert(im:max()<=1,"A: badly scaled inputs")
    assert(im:min()>=-1,"A: badly scaled inputs")

    local oW = sampleSize[2]
    local oH = sampleSize[2]
    local iH = im:size(2)
    local iW = im:size(3)
    if (opt.resize_or_crop == 'resize_and_crop' ) then
      local h1, w1 = 0, 0
      if iH~=oH then
        h1 = math.ceil(torch.uniform(1e-2, iH-oH))
      end
      if iW~=oW then
        w1 = math.ceil(torch.uniform(1e-2, iW-oW))
      end
      if iH ~= oH or iW ~= oW then
        im = image.crop(im, w1, h1, w1 + oW, h1 + oH)
      end
    elseif (opt.resize_or_crop == 'combined') then
      local sH = math.min(math.ceil(oH * torch.uniform(1+1e-2, 2.0-1e-2)), iH-1e-2)
      local sW = math.min(math.ceil(oW * torch.uniform(1+1e-2, 2.0-1e-2)), iW-1e-2)
      local h1 = math.ceil(torch.uniform(1e-2, iH-sH))
      local w1 = math.ceil(torch.uniform(1e-2, iW-sW))
      im = image.crop(im, w1, h1, w1 + sW, h1 + sH)
      im = image.scale(im, oW, oH)    elseif (opt.resize_or_crop == 'crop') then
      local w = math.min(math.min(oH, iH),iW)
      w = math.floor(w/4)*4
      local x = math.floor(torch.uniform(0, iW - w))
      local y = math.floor(torch.uniform(0, iH - w))
      im = image.crop(im, x, y, x+w, y+w)
    elseif (opt.resize_or_crop == 'scale_width') then
      w = oW
      h = torch.floor(iH * oW/iW)
      im = image.scale(im, w, h)
    elseif (opt.resize_or_crop == 'scale_height') then
      h = oH
      w = torch.floor(iW * oH / iH)
      im = image.scale(im, 448, h)
    end

    if opt.flip == 1 and torch.uniform() > 0.5 then
        im = image.hflip(im)
    end

  return im

end

