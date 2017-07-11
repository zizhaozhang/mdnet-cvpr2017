

local torchtools = {}

function torchtools.printNet(net, show_noweight_layer)

	show_noweight_layer = show_noweight_layer or false
	len = #(net.modules)
	local c = 0
	for i = 1, len do
		mod = net.modules[i]		
		if mod.weight ~= nil or show_noweight_layer then
			
			if mod.weight ~= nil then
				io.write('layer ['..i..']   ')
				if mod.name ~= nil then
					io.write ('<'..mod.name..">: ")
				else
					io.write ('<'..mod.__typename..">: ")
				end
				wz = mod.weight:size()
				for j = 1, #wz do
					io.write (wz[j]..' ')
				end
				io.write('\n')
				c = c + 1
			elseif show_noweight_layer then
				io.write('layer ['..i..']   ')
				if mod.name ~= nil then
					io.write ('<'..mod.name..">: ")
				else
					io.write ('<'..mod.__typename..">\n")
				end
			end
		end

	end
	io.write('totally '..c..' parameterized layers\n')
end
function torchtools.printBlobs(net)
	for i = 1, #(net.modules) do
		mod = net.modules[i]
		if mod.output ~= nil and type(mod.output) ~= 'table' then
			io.write('layer '..i..' ')
			if mod.name ~= nil then
				io.write ('<'..mod.name..">: ")
			else
				io.write ('<'..mod.__typename..">: ")
			end
			wz = mod.output:size()
			for j = 1, #wz do
				io.write (wz[j]..' ')
			end
			io.write('\n')
		end
	end
end
function torchtools.copy_from(src_net, target_net)

	local src_net = src_net.modules
	local target_net = target_net.modules
	local get_weight = function (net, i)
                           return net[i].weight
                       end
	local get_bias = function (net, i)
						return net[i].bias
					end
	-- layer pointers to src net
	src_table = {}
	for i = 1, #src_net do
        local name = src_net[i].name
        if name ~= nil and src_net[i].weight ~= nil then
            src_table[name] = {weights=get_weight(src_net, i), bias=get_bias(src_net,i)}
        end
	end
	local compare_size = function (a, b) local same = true
							for i=1, #a do
								same = same and a[i] == b[i]
							end
							return same end
	-- assign to target net
	local c = 0
	local m = 0
	for i = 1, #target_net do
		local name = target_net[i].name
        
		if name ~= nil and src_table[name] ~= nil then
			local weight_pt = get_weight(target_net, i)
			local bias_pt = get_bias(target_net, i)
            -- if weight_pt:size() == src_table[name]:size() then
            io.write('copy layer ['..i..']: ')
			-- assert(torch.eq(weight_pt:size(),src_table[name]:size()):sum()==#src_table[name]:size())
			io.write(name..": ")
			local sz = weight_pt:size()
				for j = 1,#sz do
					io.write(sz[j]..' ')
				end
			io.write('  type: '..weight_pt.__typename)
			if not compare_size(weight_pt:size(), src_table[name]['weights']:size()) then
				io.write('! weights size is different, skip')
				m = m +1
			else
				weight_pt:copy(src_table[name]['weights'])
				bias_pt:copy(src_table[name]['bias'])
				c = c + 1			
			end
			io.write('\n')
				
		end
	end
	print ('totally '..c..' layers are copied and '..m..' layers are skipped')
end
		
return torchtools
