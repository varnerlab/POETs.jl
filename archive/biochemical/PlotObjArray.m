hold on

% Load the ensemble files from disk -
ec_array_full = load("./data/ec_array.dat")
rank_array = load("./data/rank_array.dat")

% What is the ensemble?
idx_rank = find(rank_array==0)
ec_array_R0 = ec_array_full(:,idx_rank)

# How big is the ensemble -
[number_of_rows,number_of_cols] = size(ec_array_R0)

% # Rescale the ec_array -
% scaled_ec_array_R0 = zeros(number_of_rows,number_of_cols);
% for row_index = 1:number_of_rows
%
% 	min_value = min(ec_array_R0(row_index,:));
% 	max_value = max(ec_array_R0(row_index,:));
%
% 	for col_index = 1:number_of_cols
% 		scaled_ec_array_R0(row_index,col_index) = (ec_array_R0(row_index,col_index) - min_value)/(max_value - min_value);
% 	end
% end

% What is the ensemble?
idx_rank = find(rank_array==1)
ec_array_R1 = ec_array_full(:,idx_rank)

% # How big is the ensemble -
% [number_of_rows,number_of_cols] = size(ec_array_R1)
%
% # Rescale the ec_array -
% scaled_ec_array_R1 = zeros(number_of_rows,number_of_cols);
% for row_index = 1:number_of_rows
%
% 	min_value = min(ec_array_R1(row_index,:));
% 	max_value = max(ec_array_R1(row_index,:));
%
% 	for col_index = 1:number_of_cols
% 		scaled_ec_array_R1(row_index,col_index) = (ec_array_R1(row_index,col_index) - min_value)/(max_value - min_value);
% 	end
% end

scaled_ec_array_R0 = ec_array_R0
scaled_ec_array_R1 = ec_array_R1

% Go through the rows and cols and make the plots
offset = 0.1;
for row_index = 1:number_of_rows

  % Go through the cols -
	for col_index = 1:row_index

		% What is this index -
		INDEX = col_index+number_of_rows*(row_index-1);

		# make  plot -
		step_index_R0 = 1:2:length(ec_array_R0(1,:))
		step_index_R1 = 1:2:length(ec_array_R1(1,:))
		subplot(number_of_rows,number_of_rows,INDEX);
		plot(scaled_ec_array_R0(row_index,step_index_R0),scaled_ec_array_R0(col_index,step_index_R0),'k.',scaled_ec_array_R1(row_index,step_index_R1),scaled_ec_array_R1(col_index,step_index_R1),'.',"color",[0.7 0.7 0.7]);

		# write the axis -
		x_axis_label_string = ["O",num2str(col_index)];
		y_axis_label_string = ["O",num2str(row_index)];

		xlabel(x_axis_label_string);
		ylabel(y_axis_label_string);

		set(gca,'xtick',[])
		set(gca,'xticklabel',[])
		set(gca,'ytick',[])
		set(gca,"yticklabel",[])

	end
end
