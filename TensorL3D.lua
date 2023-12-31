local TensorL3D = {}

local function create3DTensor(dimensionArray, initialValue)
	
	local result = {}

	for dimension1 = 1, dimensionArray[1], 1 do

		result[dimension1] =  {}

		for dimension2 = 1, dimensionArray[2], 1 do

			result[dimension1][dimension2] = table.create(dimensionArray[3], initialValue)

		end

	end
	
	return result
	
end

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function applyOperation(operation, tensor1, tensor2)
	
	local result = {}
	
	for dimension1 = 1, #tensor1, 1 do
		
		result[dimension1] = {}

		for dimension2 = 1, #tensor1[dimension1], 1 do
			
			result[dimension1][dimension2] = {}

			for dimension3 = 1, #tensor1[dimension1][dimension2], 1 do

				result[dimension1][dimension2][dimension3] = operation(tensor1[dimension1][dimension2][dimension3], tensor2[dimension1][dimension2][dimension3]) 

			end

		end

	end
	
	return result
	
end

function TensorL3D.new(...)
	
	local self = setmetatable({}, TensorL3D)

	self.Values = ...

	return self
	
end

function TensorL3D.create(dimensionArray, initialValue)
	
	initialValue = initialValue or 0
	
	local self = setmetatable({}, TensorL3D)
	
	self.Values = create3DTensor(dimensionArray, initialValue)
	
	return self
	
end

function TensorL3D:broadcast(dimensionArray, values)

	local isNumber = typeof(values) == "number"

	if isNumber then return self.create(dimensionArray, values) end
	
	return values

end

function TensorL3D:getDimensionArray()
	
	return {#self, #self[1], #self[1][1]}
	
end

local function generateTensor2DString(tensor2D)

	if tensor2D == nil then return "" end

	local numberOfRows = #tensor2D

	local numberOfColumns = #tensor2D[1]

	local columnWidths = {}

	for column = 1, numberOfColumns do

		local maxWidth = 0

		for row = 1, numberOfRows do

			local cellWidth = string.len(tostring(tensor2D[row][column]))

			if (cellWidth > maxWidth) then

				maxWidth = cellWidth

			end

		end

		columnWidths[column] = maxWidth

	end

	local text = ""

	for row = 1, numberOfRows do

		text = text .. "{"

		for column = 1, numberOfColumns do

			local cellValue = tensor2D[row][column]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = columnWidths[column] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText
		end

		text = text .. " }\n"
		
	end

	return text

end

function TensorL3D:print()

	print(self)
	
end

function TensorL3D:transpose(dimension1, dimension2)
	
	if (typeof(dimension1) ~= "number") or (typeof(dimension2) ~= "number") then error("Dimensions are not numbers.") end
	
	local dimensionArray = self:getDimensionArray()
	
	local result = {}

	-- Check if the specified dimensions are within the valid range
	if (dimension1 < 1) or (dimension1 > dimensionArray[1]) or (dimension2 < 1) or (dimension2 > dimensionArray[1]) or (dimension1 == dimension2) then
		error("Invalid dimensions for transpose.")
	end

	-- Initialize the transposed tensor with the same dimensions as the input tensor
	for i = 1, dimensionArray[1] do
		
		result[i] = {}
		
		for j = 1, #self[i] do
			
			result[i][j] = {}
			
		end
		
	end

	-- Perform the transpose operation
	for i = 1, dimensionArray[1] do
		
		for j = 1, #self[i] do
			
			for k = 1, #self[i][j] do
				
				result[i][j][k] = self[i][j][k]
				
			end
			
		end
		
	end

	-- Swap the specified dimensions
	for i = 1, dimensionArray[1] do
		
		for j = 1, #self[i] do
			
			for k = 1, #self[i][j] do
				
				if dimension1 ~= i and dimension2 ~= i then
					
					result[i][j][k] = self[i][j][k]
					
				elseif dimension1 == i then
					
					result[dimension2][j][k] = self[i][j][k]
					
				elseif dimension2 == i then
					
					result[dimension1][j][k] = self[i][j][k]
					
				end
				
			end
			
		end
		
	end

	return self.new(result)
	
end

function TensorL3D:__eq(other)
	
	local success = pcall(function() local _ = other[1][1][1] end)
	
	if not success then return false end
	
	for dimension1 = 1, #self, 1 do

		for dimension2 = 1, #self[dimension1], 1 do

			for dimension3 = 1, #self[dimension1][dimension2], 1 do

				if (self[dimension1][dimension2][dimension3] ~= other[dimension1][dimension2][dimension3]) then return false end

			end

		end

	end

	return true
	
end

function TensorL3D:isEqualTo(other)

	local success = pcall(function() local _ = other[1][1][1] end)

	if not success then return error("The other value is not a tensor.") end

	local operation = function(a, b) return (a == b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function TensorL3D:isGreaterThan(other)

	local success = pcall(function() local _ = other[1][1][1] end)

	if not success then return error("The other value is not a tensor.") end

	local operation = function(a, b) return (a > b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function TensorL3D:isGreaterOrEqualTo(other)

	local success = pcall(function() local _ = other[1][1][1] end)

	if not success then return error("The other value is not a tensor.") end

	local operation = function(a, b) return (a >= b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function TensorL3D:isLessThan(other)

	local success = pcall(function() local _ = other[1][1][1] end)

	if not success then return error("The other value is not a tensor.") end
	
	local operation = function(a, b) return (a < b) end

	local result = applyOperation(operation, self, other)
	
	return self.new(result)

end

function TensorL3D:isLessOrEqualTo(other)

	local success = pcall(function() local _ = other[1][1][1] end)

	if not success then return error("The other value is not a tensor.") end

	local operation = function(a, b) return (a <= b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function TensorL3D:sum(dimension)
	
	local dimensionArray = self:getDimensionArray()

	local newDimensionArray = deepCopyTable(dimensionArray)

	if dimension then newDimensionArray[dimension] = 1 end

	local result = (not dimension and 0) or self.create(newDimensionArray, 0)

	for dimension1 = 1, dimensionArray[1], 1 do

		for dimension2 = 1, dimensionArray[2], 1 do

			for dimension3 = 1, dimensionArray[3], 1 do
				
				if (dimension == nil) then
					
					result += self[dimension1][dimension2][dimension3]
				
				elseif (dimension == 1) then

					result[1][dimension2][dimension3] += self[dimension1][dimension2][dimension3]	

				elseif (dimension == 2) then

					result[dimension1][1][dimension3] += self[dimension1][dimension2][dimension3]

				elseif (dimension == 3) then

					result[dimension1][dimension2][1] += self[dimension1][dimension2][dimension3]
					
				else
					
					error("Invalid dimension.")

				end 

			end

		end	

	end
	
	return result
	
end

function TensorL3D:tensorProduct(other)
	
	local success = pcall(function() local _ = other[1][1][1] end)

	if not success then return error("The other value is not a tensor.") end
	
	local dimensionArray = self:getDimensionArray()
	
	local otherDimensionArray = other:getDimensionArray()
	
	for index, _ in ipairs(dimensionArray) do if (dimensionArray[index] ~= otherDimensionArray[index]) then error("Tensors are not the same size!") end end

	local result = 0
	
	for dimension1 = 1, dimensionArray[1] do
		
		for dimension2 = 1, dimensionArray[2] do
			
			for dimension3 = 1, dimensionArray[3] do
				
				result = result + self[dimension1][dimension2][dimension3] * other[dimension1][dimension2][dimension3]
				
			end
			
		end
		
	end
	
	return result
	
end

function TensorL3D:innerProduct(other)

	local success = pcall(function() local _ = other[1][1][1] end)

	if not success then return error("The other value is not a tensor.") end

	local dimensionArray = self:getDimensionArray()

	local otherDimensionArray = other:getDimensionArray()
	
	if (dimensionArray[1] ~= otherDimensionArray[2]) then error("Tensors must have the same shape for inner product.") end

	for index, _ in ipairs(dimensionArray) do if (dimensionArray[index] ~= otherDimensionArray[index]) then error("Tensors are not the same size!") end end

	local result = 0

	for dimension1 = 1, dimensionArray[1] do

		for dimension2 = 1, dimensionArray[2] do

			for dimension3 = 1, dimensionArray[3] do

				result = result + self[dimension1][dimension2][dimension3] * other[dimension1][dimension2][dimension3]

			end

		end

	end

	return result

end

function TensorL3D:outerProduct(other)

	local success = pcall(function() local _ = other[1][1][1] end)

	if not success then return error("The other value is not a tensor.") end

	local dimensionArray = self:getDimensionArray()

	local otherDimensionArray = other:getDimensionArray()

	if (dimensionArray[1] ~= otherDimensionArray[2]) then error("Tensors must have the same shape for inner product.") end

	for index, _ in ipairs(dimensionArray) do if (dimensionArray[index] ~= otherDimensionArray[index]) then error("Tensors are not the same size!") end end

	local result = {}

	for dimension1 = 1, dimensionArray[1] do
		
		result[dimension1] = {}
		
		for dimension2 = 1, #self[dimension1] do
			
			result[dimension1][dimension2] = {}
			
			for dimension3 = 1, #self[dimension1][dimension2] do
				
				result[dimension1][dimension2][dimension3] = self[dimension1][dimension2][dimension3] * other[dimension1][dimension2][dimension3]
				
			end
			
		end
		
	end

	return self.new(result)

end

function TensorL3D:copy()
	
	return deepCopyTable(self)
	
end

function TensorL3D:rawCopy()

	return deepCopyTable(self.Values)

end

function TensorL3D:applyFunction(functionToApply, ...)

	local tensorValues

	local tensors = {...}
	
	local dimensionArray = self:getDimensionArray()

	local result = self.create(table.unpack(dimensionArray))

	for dimension1 = 1, dimensionArray[1], 1 do

		for dimension2 = 1, dimensionArray[2], 1 do

			tensorValues = {}
			
			for dimension3 = 1, dimensionArray[3], 1 do
				
				for matrixArgument = 1, #tensors, 1  do

					table.insert(tensorValues, tensors[matrixArgument][dimension1][dimension2][dimension3])

				end 
				
				result[dimension1][dimension2][dimension3] = functionToApply(table.unpack(tensorValues))
				
			end
			
		end	

	end

	return result

end

function TensorL3D:__add(other)
	
	local other = self:broadcast(self:getDimensionArray(), other)
	
	local operation = function(a, b) return (a + b) end
	
	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function TensorL3D:__sub(other)
	
	local other = self:broadcast(self:getDimensionArray(), other)

	local operation = function(a, b) return (a - b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function TensorL3D:__mul(other)
	
	local other = self:broadcast(self:getDimensionArray(), other)

	local operation = function(a, b) return (a * b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function TensorL3D:__div(other)
	
	local other = self:broadcast(self:getDimensionArray(), other)

	local operation = function(a, b) return (a / b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function TensorL3D:__unm(other)
	
	local result = deepCopyTable(self)
	
	for dimension1 = 1, #self, 1 do

		for dimension2 = 1, #self[dimension1], 1 do
			
			for dimension3 = 1, #self[dimension1][dimension2], 1 do

				result[dimension1][dimension2][dimension3] *= -1

			end

		end

	end
	
	return result
	
end

function TensorL3D:__tostring()
	
	local text = "\n\n{\n\n"

	local generatedText

	for index = 1, #self, 1 do

		generatedText = generateTensor2DString(self[index])

		text = text .. generatedText .. "\n"

	end

	text = text .. "}\n\n"

	return text
	
end

function TensorL3D:__len()
	
	return #self.Values
	
end

function TensorL3D:__index(index)
	
	if (typeof(index) == "number") then
		
		return rawget(self.Values, index)
		
	else
		
		return rawget(TensorL3D, index)
		
	end
end

function TensorL3D:__newindex(index, value)
	
	rawset(self, index, value)
	
end

return TensorL3D
