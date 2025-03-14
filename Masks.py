import numpy as np
import cv2



'''
	The first element of every value represents x_min, x_max, y_min, y_max of the original numpy array and
	the following elements are (x1, y1, x2, y2) - coordinates of the top-left and bottom-right corners).  
'''
masks = {
	'FCGB1': [
		[np.float64(-825.0), np.float64(825.0), np.float64(-1525.0), np.float64(525.0)],
		[np.float64(-825.0), np.float64(-375.0), np.float64(825.0), np.float64(-625.0)], 
		[np.float64(-125.0), np.float64(-625.0), np.float64(125.0), np.float64(-1525.0)]
	],
	'FCGB2': [
		[np.float64(-615.0), np.float64(615.0), np.float64(-970.0), np.float64(630.0)], 
		[np.float64(-615.0), np.float64(-185.0), np.float64(615.0), np.float64(-485.0)],
		[np.float64(-150.0), np.float64(-485.0), np.float64(150.0), np.float64(-970.0)],
	],
	'FCGB3': [
		[np.float64(-615.0), np.float64(615.0), np.float64(-985.0), np.float64(615.0)],
		[np.float64(-150.0), np.float64(599.48), np.float64(150.0), np.float64(-985.0)],
	],
	'FCSP2': [
		[np.float64(-750.0), np.float64(750.0), np.float64(-750.0), np.float64(750.0)], 
		[np.float64(-150.0), np.float64(733.49), np.float64(150.0), np.float64(-750.0)],
	],
	'FCSP3': [
		[np.float64(-850.0), np.float64(850.0), np.float64(-1485.0), np.float64(765.0)], 
		[np.float64(-850.0), np.float64(-135.0), np.float64(850.0), np.float64(-585.0)],
		[np.float64(-150.0), np.float64(-585.0), np.float64(150.0), np.float64(-1485.0)],
	],
	'FCSP4': [
		[np.float64(-705.0), np.float64(705.0), np.float64(-975.0), np.float64(705.0)],
		[np.float64(-705.0), np.float64(-135.0), np.float64(705.0), np.float64(-385.0)],
		[np.float64(-135.0), np.float64(-385.0), np.float64(135.0), np.float64(-975.0)],
	],
	'VTF2': [
		[np.float64(-670.0), np.float64(670.0), np.float64(-1120.0), np.float64(610.0)],
		[np.float64(-670.0), np.float64(10.0), np.float64(670.0), np.float64(-250.0)],
		[np.float64(-130.0), np.float64(-250.0), np.float64(130.0), np.float64(-1120.0)],
	],
}

def apply_mask_on_2Darray(array: np.ndarray, mask: list[list[float]]):
	# raise error if input array is not 2D
	if array.ndim != 2:
		raise ValueError("The input array must be 2D.")
	
	# get the dimensions of the array
	resized_dim = array.shape

	# extract data from mask
	(data_x_min, data_x_max, data_y_min, data_y_max), *rectangles = mask

	# find the scale in each axis
	x_range, y_range = data_x_max - data_x_min, data_y_max - data_y_min
	x_scale = resized_dim[0] / x_range
	y_scale = resized_dim[1] / y_range

	# Loop through the original rectangles and scale them
	for (x1, y1, x2, y2) in rectangles:
		# print(resized_image.shape, (x1, x2, y1, y2))

		# Move the origin of the rectangle to the top-left
		x1 -= data_x_min
		x2 -= data_x_min
		y1 -= data_y_min
		y2 -= data_y_min

		# Scale the dimensions to resized_dim
		scaled_x1 = x1 * x_scale
		scaled_x2 = x2 * x_scale
		scaled_y1 = y1 * y_scale
		scaled_y2 = y2 * y_scale

		# Ensure valid rectangle dimensions
		scaled_x1, scaled_x2 = sorted([scaled_x1, scaled_x2])
		scaled_y1, scaled_y2 = sorted([scaled_y1, scaled_y2])

		# Overwrite the values in the rectangle region with 0 - black
		array[int(scaled_y1):int(scaled_y2), int(scaled_x1):int(scaled_x2)] = 0

		# print(resized_image.shape, (scaled_x1, scaled_y1, scaled_x2, scaled_y2))
		
	return array













if __name__ == "__main__":
	from interpolated_greyscale_image_generator import load_npy_file, process_and_group_data, interpolate_nlcreq, save_normalized_greyscale_image

	arr = load_npy_file(r'.\data\SAC105\FCGB1_sac105.npy')
	grouped_data = process_and_group_data(arr)
	NLCREQ_interp = interpolate_nlcreq(grouped_data, grid_size=100)
	masked_NLCREQ_interp = apply_mask_on_2Darray(NLCREQ_interp, masks['FCGB1'])
	save_normalized_greyscale_image(masked_NLCREQ_interp, r'.\images\masked.png')