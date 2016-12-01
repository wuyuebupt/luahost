--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

-- local t = require '../datasets/transforms'
local t = require './transforms'


-- model 1 
local imagenetLabel1 = require './model10k_1st/imagenet'
local model1 = torch.load('./model10k_1st/model_best_clean.t7')
-- local softMaxLayer1 = cudnn.SoftMax():cuda()
-- add Softmax layer
-- model1:add(softMaxLayer1)
-- model1:evaluate()

-- model 2
local imagenetLabel2 = require './model10k_2nd/imagenet'
local model2 = torch.load('./model10k_2nd/model_best_clean.t7')
-- local softMaxLayer2 = cudnn.SoftMax():cuda()
-- add Softmax layer
-- model2:add(softMaxLayer2)
-- model2:evaluate()

-- model 3 
local imagenetLabel3 = require './model10k_3rd/imagenet'
local model3 = torch.load('./model10k_3rd/model_best_clean.t7')
-- local softMaxLayer3 = cudnn.SoftMax():cuda()
-- add Softmax layer
-- model3:add(softMaxLayer3)
-- model3:evaluate()

-- model 4
local imagenetLabel4 = require './model10k_4th/imagenet'
local model4 = torch.load('./model10k_4th/model_best_clean.t7')
-- local softMaxLayer4 = cudnn.SoftMax():cuda()
-- add Softmax layer
-- model4:add(softMaxLayer4)
-- model4:evaluate()

-- model 5 
local imagenetLabel5 = require './model10k_5th/imagenet'
local model5 = torch.load('./model10k_5th/model_best_clean.t7')
-- local softMaxLayer5 = cudnn.SoftMax():cuda()
-- add Softmax layer
-- model5:add(softMaxLayer5)
-- model5:evaluate()

local imagenetLabel6 = require './model100k_18layer/imagenet'
local model6 = torch.load('./model100k_18layer/model_3_clean.t7')

local threshold = 0.95

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.TwoCrop(224),
}


function file_exists(file)
	local f = io.open(file, "rb")
	if f then f:close() end
	return f ~= nil
end

function lines_from(file)
	if not file_exists(file) then return {} end
	lines = {}
	for line in io.lines(file) do 
		lines[#lines + 1] = line
	end
	return lines
end


-- take one line as test
local file = arg[1]
local lines = lines_from(file)


local N = 5


-- for sort
function spairs(t, order)
	local keys = {}
	for k in pairs(t) do keys[#keys+1] = k end

	if order then
		table.sort(keys, function(a,b) return order(t, a, b) end)
	else
		table.sort(keys)
	
	end

	local i = 0
	return function()
		i = i + 1
		if keys[i] then
			return keys[i], t[keys[i]]
		end
	end
end

function compare(t,a,b) 
	return t[b] < t[a] 
end

-- for i=2,#arg do
for k,v in pairs(lines) do
   -- load the image as a RGB float tensor with values 0..1
   -- local img = image.load(arg[i], 3, 'float')
   local begintime = os.clock()

   -- get batch 
   local img = image.load(v, 3, 'float')
   img = transform(img)
   local imageSize = img:size():totable()
   table.remove(imageSize, 1)
   local batch = img:view(2, table.unpack(imageSize)):cuda()
   -- local batch = img:view(1, table.unpack(img:size():totable()))
   -- local batch = img:view(1, table.unpack(img:size():totable()))

   -- Get the output of the softmax for model 1 to 5
   local allres = {}

   local output1 = model1:forward(batch):squeeze()
   output1 = output1:view(output1:size(1)/2, 2, output1:size(2)):sum(2):squeeze():div(2)
   local probs1, indexes1 = output1:topk(N, true, true)
   for n=1,N do
	   allres[ imagenetLabel1[indexes1[n]]] = probs1[n]
   end
   
   local output2 = model2:forward(batch):squeeze()
   output2 = output2:view(output2:size(1)/2, 2, output2:size(2)):sum(2):squeeze():div(2)
   local probs2, indexes2 = output2:topk(N, true, true)
   for n=1,N do
	   allres[ imagenetLabel2[indexes2[n]]] = probs2[n]
   end

   local output3 = model3:forward(batch):squeeze()
   output3 = output3:view(output3:size(1)/2, 2, output3:size(2)):sum(2):squeeze():div(2)
   local probs3, indexes3 = output3:topk(N, true, true)
   for n=1,N do
	   allres[ imagenetLabel3[indexes3[n]]] = probs3[n]
   end

   local output4 = model4:forward(batch):squeeze()
   output4 = output4:view(output4:size(1)/2, 2, output4:size(2)):sum(2):squeeze():div(2)
   local probs4, indexes4 = output4:topk(N, true, true)
   for n=1,N do
	   allres[ imagenetLabel4[indexes4[n]]] = probs4[n]
   end

   local output5 = model5:forward(batch):squeeze()
   output5 = output5:view(output5:size(1)/2, 2, output5:size(2)):sum(2):squeeze():div(2)
   local probs5, indexes5 = output5:topk(N, true, true)
   for n=1,N do
	   allres[ imagenetLabel5[indexes5[n]]] = probs5[n]
   end



   local allres_6 = {}
   local output6 = model6:forward(batch):squeeze()
   output6 = output6:view(output6:size(1)/2, 2, output6:size(2)):sum(2):squeeze():div(2)
   local probs6, indexes6 = output6:topk(N, true, true)

   for n=1,N do
	   local newprob = 0.30206/(1+math.exp(-4.2523*probs6[n])) + 0.70097*math.tanh(13.44*probs6[n])
	   allres_6[ imagenetLabel6[indexes6[n]]] = newprob
   end
   local topprob_6 = allres_6[imagenetLabel6[indexes6[1]]]


   for k,v in spairs(allres, compare) do
       print(k,v)
   end

   local count = 1
   outstr = ""

   if topprob_6 > threshold then
	   -- update with model 6
	   for k,v in spairs(allres_6, compare) do
		   if count < N then
			   outstr = outstr .. k .. ':' .. v .. ';';
		   else
			   outstr = outstr .. k .. ':' .. v;
			   break;
		   end
		   count = count + 1
	   end
   else
	   for k,v in spairs(allres, compare) do
		   print(k,v)
		   if count < N then
			   outstr = outstr .. k .. ':' .. v .. ';';
		   else
			   outstr = outstr .. k .. ':' .. v;
			   break;
		   end
		   count = count + 1
	   end
   end
   print(outstr)

   local endtime = os.clock()
   print(string.format("forward time: %.2f\n", endtime- begintime))
  end

local uv = require('luv')


local function create_server(host, port, on_connection)
	local server = uv.new_tcp()
	server:bind(host, port)
	server:listen(128, function(err)
		assert(not err, err)
		local client = uv.new_tcp()
		server:accept(client)
		on_connection(client)
	end)
	return server
end



local server = create_server("127.0.0.1", 9999, function (client)
	client:read_start(function (err, chunk)
		 assert(not err, err)
		 if chunk then
			 print(chunk)

   local begintime = os.clock()
   -- get batch 
   local img = image.load(chunk, 3, 'float')
   img = transform(img)
   local imageSize = img:size():totable()
   table.remove(imageSize, 1)
   local batch = img:view(2, table.unpack(imageSize)):cuda()
   -- local batch = img:view(1, table.unpack(img:size():totable()))
   -- local batch = img:view(1, table.unpack(img:size():totable()))

   -- Get the output of the softmax for model 1 to 5
   local allres = {}

   local output1 = model1:forward(batch):squeeze()
   output1 = output1:view(output1:size(1)/2, 2, output1:size(2)):sum(2):squeeze():div(2)
   local probs1, indexes1 = output1:topk(N, true, true)
   for n=1,N do
	   allres[ imagenetLabel1[indexes1[n]]] = probs1[n]
   end
   
   local output2 = model2:forward(batch):squeeze()
   output2 = output2:view(output2:size(1)/2, 2, output2:size(2)):sum(2):squeeze():div(2)
   local probs2, indexes2 = output2:topk(N, true, true)
   for n=1,N do
	   allres[ imagenetLabel2[indexes2[n]]] = probs2[n]
   end

   local output3 = model3:forward(batch):squeeze()
   output3 = output3:view(output3:size(1)/2, 2, output3:size(2)):sum(2):squeeze():div(2)
   local probs3, indexes3 = output3:topk(N, true, true)
   for n=1,N do
	   allres[ imagenetLabel3[indexes3[n]]] = probs3[n]
   end

   local output4 = model4:forward(batch):squeeze()
   output4 = output4:view(output4:size(1)/2, 2, output4:size(2)):sum(2):squeeze():div(2)
   local probs4, indexes4 = output4:topk(N, true, true)
   for n=1,N do
	   allres[ imagenetLabel4[indexes4[n]]] = probs4[n]
   end

   local output5 = model5:forward(batch):squeeze()
   output5 = output5:view(output5:size(1)/2, 2, output5:size(2)):sum(2):squeeze():div(2)
   local probs5, indexes5 = output5:topk(N, true, true)
   for n=1,N do
	   allres[ imagenetLabel5[indexes5[n]]] = probs5[n]
   end

   local allres_6 = {}
   local output6 = model6:forward(batch):squeeze()
   output6 = output6:view(output6:size(1)/2, 2, output6:size(2)):sum(2):squeeze():div(2)
   local probs6, indexes6 = output6:topk(N, true, true)

   for n=1,N do
	   local newprob = 0.30206/(1+math.exp(-4.2523*probs6[n])) + 0.70097*math.tanh(13.44*probs6[n])
	   allres_6[ imagenetLabel6[indexes6[n]]] = newprob
   end
   local topprob_6 = allres_6[imagenetLabel6[indexes6[1]]]


   local count = 1
   outstr = ""
   if topprob_6 > threshold then
	   -- update with model 6
	   for k,v in spairs(allres_6, compare) do
		   if count < N then
			   outstr = outstr .. k .. ':' .. v .. ';';
		   else
			   outstr = outstr .. k .. ':' .. v;
			   break;
		   end
		   count = count + 1
	   end
   else
	   for k,v in spairs(allres, compare) do
		   if count < N then
			   outstr = outstr .. k .. ':' .. v .. ';';
		   else
			   outstr = outstr .. k .. ':' .. v;
			   break;
		   end
		   count = count + 1
	   end
   end

   print(outstr)
   local endtime = os.clock()
   print(string.format("forward time: %.2f\n", endtime- begintime))
			 client:write(outstr)
			 client:close()
		 else
			 client:close()
		 end
	 end)
 end)

print("TCP Echo serverr listening on port " .. server:getsockname().port)
uv.run()


