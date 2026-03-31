using PyPlot

# Load and plot the data -
species_index = 4
MEASURED_ARRAY_1 = readdlm("./data/MEASUREMENT_SET_1.dat")
MEASURED_ARRAY_2 = readdlm("./data/MEASUREMENT_SET_2.dat")
MEASURED_ARRAY_3 = readdlm("./data/MEASUREMENT_SET_3.dat")
MEASURED_ARRAY_4 = readdlm("./data/MEASUREMENT_SET_4.dat")

skip = 40
idx_range = 1:skip:length(MEASURED_ARRAY_1[:,1])
time_array = MEASURED_ARRAY_1[idx_range,1]
data_array_1 = MEASURED_ARRAY_1[idx_range,species_index]
data_array_2 = MEASURED_ARRAY_2[idx_range,species_index]
data_array_3 = MEASURED_ARRAY_3[idx_range,species_index]
data_array_4 = MEASURED_ARRAY_4[idx_range,species_index]

# cache -
data_cache = zeros(length(idx_range),4)
for index in collect(1:length(idx_range))
  data_cache[index,1] = data_array_1[index]
  data_cache[index,2] = data_array_2[index]
  data_cache[index,3] = data_array_3[index]
  data_cache[index,4] = data_array_4[index]
end

# calculate the mean, and std -
mean_value = mean(data_cache,2)
std_value = std(data_cache,2)
SF = (1.96/sqrt(length(mean_value)))
plot(time_array,mean_value,"ko")

range_array = [SF*transpose(std_value) ; SF*transpose(std_value)]
errorbar(time_array,mean_value, yerr=range_array,fmt="ko")
