import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
from tqdm import tqdm
from typing import Sequence, Optional
from Masks import masks, apply_mask_on_2Darray

# Define needed functions
def load_npy_file(npy_file: str) -> np.ndarray:
    """
    Loads a .npy file and handles potential errors during the loading process.
    
    Parameters:
        npy_file (str): The path to the .npy file to be loaded.
    
    Returns:
        np.ndarray: The loaded numpy array if successful, otherwise None.
    """
    try:
        arr = np.load(npy_file)
        return arr
    
    except IOError as e:
        print(f"Failed to read the file: {npy_file}")
        print(f"Error message: {str(e)}")
        return None
    
    except Exception as e:
        print(f"An error occurred while reading the file: {npy_file}")
        print(f"Error message: {str(e)}")
        return None

def process_and_group_data(arr: np.ndarray, debug: bool = False) -> pd.DataFrame:
	"""
	Creates a DataFrame from a numpy array, groups by X and Y Location, and calculates the mean of NLCREQ.
	
	Parameters:
		arr (np.ndarray): 2D numpy array with columns ['X Location (µm)', 'Y Location (µm)', 'Z Location (µm)', 'NLCREQ (µm/µm)'].
		debug (bool): If True, it will plot the X-Y scatter plot before and after grouping the data.
	
	Returns:
		pd.DataFrame: Grouped data with averaged NLCREQ values.
	"""
	# Create DataFrame from numpy array
	df = pd.DataFrame(arr, columns=['X Location (µm)', 'Y Location (µm)', 'Z Location (µm)', 'NLCREQ (µm/µm)'])
	
	# Plot X-Y plane before grouping if debug is True
	if debug:
		plot_xy_plane(df['X Location (µm)'], df['Y Location (µm)'], x_label='X Location (µm)', y_label='Y Location (µm)', title=None)

	# Group by X and Y Location and calculate the mean of NLCREQ
	grouped_data = df.groupby(['X Location (µm)', 'Y Location (µm)'])['NLCREQ (µm/µm)'].mean().reset_index()
	
	# Round the NLCREQ values to 4 decimal places
	grouped_data['NLCREQ (µm/µm)'] = grouped_data['NLCREQ (µm/µm)'].round(4)

	# Plot X-Y plane after grouping if debug is True
	if debug:
		plot_xy_plane(grouped_data['X Location (µm)'], grouped_data['Y Location (µm)'],
					c=grouped_data['NLCREQ (µm/µm)'], x_label='X Location (µm)', y_label='Y Location (µm)', 
					colorbar_label='NLCREQ (µm/µm)', title='Scatterplot - XY (NLCREQ averaged)')

	return grouped_data

def plot_xy_plane(
				X: Sequence[float], 
				Y: Sequence[float], 
				c: Optional[Sequence[float]] = None, 
				cmap: str = 'viridis', 
				x_label: str = 'X axis', 
				y_label: str = 'Y axis',
                colorbar_label: str = 'Color Intensity',
				title: str = 'XY Plane'
			) -> None:
    plt.figure(figsize=(8, 6))
    
    # Check if color data is provided
    if c is not None:
        plt.scatter(X, Y, c=c, cmap=cmap, s=10)  # Use color mapping
        plt.colorbar(label=colorbar_label)  # Add a colorbar
    else:
        plt.plot(X, Y, 'bo', markersize=3)  # Simple scatter plot
    
    # Set labels, title, and grid
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    
    # Show the plot
    plt.show()

def interpolate_nlcreq(grouped_data: pd.DataFrame, grid_size: int, debug: bool = False) -> np.ndarray:
    """
    Interpolates the NLCREQ values onto a regular grid based on X and Y location coordinates.
    
    Parameters:
        grouped_data (pd.DataFrame): The grouped DataFrame with columns ['X Location (µm)', 'Y Location (µm)', 'NLCREQ (µm/µm)'].
        grid_size (int): The size of the grid (number of points along each axis).
        debug (bool): If True, it will plot the interpolated grid.
    
    Returns:
        np.ndarray: The interpolated NLCREQ values on the grid.
    """
    # Define grid points based on the current file's data
    x_vals = np.linspace(grouped_data['X Location (µm)'].min(), grouped_data['X Location (µm)'].max(), grid_size)
    y_vals = np.linspace(grouped_data['Y Location (µm)'].min(), grouped_data['Y Location (µm)'].max(), grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Interpolate NLCREQ values onto the grid
    NLCREQ_interp = griddata(
        (grouped_data['X Location (µm)'], grouped_data['Y Location (µm)']),
        grouped_data['NLCREQ (µm/µm)'],
        (X, Y),
        method='linear'
    )

    # Handle NaN values in the interpolated array
    NLCREQ_interp = np.nan_to_num(NLCREQ_interp)

    # Optionally plot the interpolated grid if debug is True
    if debug:
        plot_interpolated_grid(grouped_data, NLCREQ_interp)

    return NLCREQ_interp

def plot_interpolated_grid(averaged_df: pd.DataFrame, interpolated_df: np.ndarray):
	# Plotting the Averaged data and interpolated data side by side
	fig, axes = plt.subplots(1, 2, figsize=(20, 10))
	
	# Get the min and max for setting the same scale
	x_min, x_max = averaged_df['X Location (µm)'].min(), averaged_df['X Location (µm)'].max()
	y_min, y_max = averaged_df['Y Location (µm)'].min(), averaged_df['Y Location (µm)'].max()
	nlcreq_min, nlcreq_max = averaged_df['NLCREQ (µm/µm)'].min(), averaged_df['NLCREQ (µm/µm)'].max()

	# Averaged data plot
	axes[0].scatter(averaged_df['X Location (µm)'], averaged_df['Y Location (µm)'],
								c=averaged_df['NLCREQ (µm/µm)'], cmap='viridis', vmin=nlcreq_min, vmax=nlcreq_max)
	axes[0].set_title('Averaged NLCREQ Data')
	axes[0].set_xticks([])
	axes[0].set_yticks([])
	axes[0].set_xlim(x_min, x_max)
	axes[0].set_ylim(y_min, y_max)

	# plot it
	img = axes[1].imshow(interpolated_df,
							extent=(x_min, x_max, y_min, y_max),
							origin='lower',
							aspect='auto',
							cmap='viridis', vmin=nlcreq_min, vmax=nlcreq_max)
	axes[1].set_title('Interpolated NLCREQ Data')
	axes[1].set_xticks([])
	axes[1].set_yticks([])
	# axes[1].invert_yaxis()  # Invert the y-axis

	# Add a colorbar to the right side of the plots
	fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='NLCREQ (µm/µm)')

	# Display the plots
	plt.show()

def save_normalized_greyscale_image(image_array: np.ndarray, greyscale_image_save_path: str, dpi: int = 600, success_message=False) -> None:
	# Ensure the image has 2 dimensions (x, y)
    if len(image_array.shape) != 2:
        raise ValueError("Input image must be a 2-dimensional array (height, width).")
	
    # Normalize the image to the range [0, 255]
    normalized_max = np.max(image_array)
    if normalized_max == 0:
        normalized_img = np.zeros_like(image_array, dtype=np.uint8)
    else:
        normalized_img = (image_array / normalized_max * 255).astype(np.uint8)

    # Save the normalized image using Matplotlib library
    try:
        plt.imsave(greyscale_image_save_path, normalized_img, cmap='gray', format='png', dpi=dpi, origin='lower')
        if success_message: print(f"Image saved: {greyscale_image_save_path}")
    except Exception as e:
        print(f"Failed to save image at '{greyscale_image_save_path}': {e}")



def generate_images(
		npy_folders_path: Sequence[str] = [
			r'.\data\SAC105',
			r'.\data\SAC305',
			r'.\data\Senju',
		],
		grid_size: int=15000, 
		image_folder_path: str=r'.\images',
		debug=False) -> None:
	
	# Few validation steps
	for npy_folder_path in npy_folders_path:
		if not os.path.exists(npy_folder_path):
			print(f"Error: The source folder for .npy files does not exist: '{npy_folder_path}'")
			return
	if not os.path.exists(image_folder_path):
		print(f"Error: The destination folder to save images does not exist: '{image_folder_path}'")
		return


	# define variables to show the progress
	total_folders = len(npy_folders_path)
	current_folder_count = 1


	for npy_folder_path in npy_folders_path:
		# print the progress message
		print(f"Processing folder ({current_folder_count}/{total_folders})")
		print(f"Source: {npy_folder_path}")
		print(f"Destination: {os.path.join(image_folder_path, os.path.basename(npy_folder_path))}")

		# create the same folder if it doesn't exist to save the images
		os.makedirs(os.path.join(image_folder_path, os.path.basename(npy_folder_path)), exist_ok=True)

		# Get a list of all .npy files in the folder
		npy_files = [os.path.join(npy_folder_path, file) for file in os.listdir(npy_folder_path) if file.endswith('.npy')]

		# loop through each .npy file in the folder and create images for it
		for npy_file in tqdm(npy_files):
			# Create path to save the greyscale image in the image_folder_path
			greyscale_image_save_path = os.path.join(image_folder_path, os.path.basename(npy_folder_path), f'grayscale_{os.path.basename(npy_file)[:-4]}_{grid_size}x{grid_size}.png')



			# step 1: Load the numpy array from .npy file
			arr = load_npy_file(npy_file)

			# step 2: Create the dataframe and group by X and Y Location
			grouped_data = process_and_group_data(arr, debug=debug)

			# step 3: do interpolation
			NLCREQ_interp = interpolate_nlcreq(grouped_data, grid_size, debug)
            
			# step 4: apply mask
			pad_design_name = os.path.basename(npy_file).split('_')[0]
			mask = masks[pad_design_name]
			# apply the mask
			NLCREQ_interp = apply_mask_on_2Darray(NLCREQ_interp, mask)

			# Step 5: Normalize and save the final data
			save_normalized_greyscale_image(NLCREQ_interp, greyscale_image_save_path, dpi=600)

		print()
		current_folder_count += 1


if __name__ == "__main__":
	generate_images(grid_size=10)