import numpy as np



'''
	The first element of every value represents x_min, x_max, y_min, y_max of the original numpy array
	while the following elements correspond to the (x, y) coordinates of the  
	top-left and bottom-right corners, respectively means (x1, y1, x2, y2).  
'''
masks_original_dim = {
	'FCGB1_mask': [
		[np.float64(-825.0), np.float64(825.0), np.float64(-1525.0), np.float64(525.0)],
		[np.float64(-825.0), np.float64(-375.0), np.float64(825.0), np.float64(-625.0)], 
		[np.float64(-125.0), np.float64(-625.0), np.float64(125.0), np.float64(-1525.0)]
	],
	'FCGB2_mask': [
		[np.float64(-615.0), np.float64(615.0), np.float64(-970.0), np.float64(630.0)], 
		[np.float64(-615.0), np.float64(-185.0), np.float64(615.0), np.float64(-485.0)],
		[np.float64(-150.0), np.float64(-485.0), np.float64(150.0), np.float64(-970.0)],
	],
	'FCGB3_mask': [
		[np.float64(-615.0), np.float64(615.0), np.float64(-985.0), np.float64(615.0)],
		[np.float64(-150.0), np.float64(599.48), np.float64(150.0), np.float64(-985.0)],
	],
	'FCSP2_mask': [
		[np.float64(-750.0), np.float64(750.0), np.float64(-750.0), np.float64(750.0)], 
		[np.float64(-150.0), np.float64(733.49), np.float64(150.0), np.float64(-750.0)],
	],
	'FCSP3_mask': [
		[np.float64(-850.0), np.float64(850.0), np.float64(-1485.0), np.float64(765.0)], 
		[np.float64(-850.0), np.float64(-135.0), np.float64(850.0), np.float64(-585.0)],
		[np.float64(-150.0), np.float64(-585.0), np.float64(150.0), np.float64(-1485.0)],
	],
	'FCSP4_mask': [
		[np.float64(-705.0), np.float64(705.0), np.float64(-975.0), np.float64(705.0)],
		[np.float64(-705.0), np.float64(-135.0), np.float64(705.0), np.float64(-385.0)],
		[np.float64(-135.0), np.float64(-385.0), np.float64(135.0), np.float64(-975.0)],
	],
	'VTF2_mask': [
		[np.float64(-670.0), np.float64(670.0), np.float64(-1120.0), np.float64(610.0)],
		[np.float64(-670.0), np.float64(10.0), np.float64(670.0), np.float64(-250.0)],
		[np.float64(-130.0), np.float64(-250.0), np.float64(130.0), np.float64(-1120.0)],
	],
}
