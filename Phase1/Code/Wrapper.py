#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s):
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import os, cv2, skimage.transform

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

######################### Difference of Gaussian Filter Bank #########################
def convolve(image, kernel):
	"""
    Perform a 2D convolution.

    :param image: Input image or filter.
    :type image: numpy.ndarray
    :param kernel: Convolution kernel.
    :type kernel: numpy.ndarray
    :return: Convolved image.
    :rtype: numpy.ndarray
    """
	# Kernel flip for convolution
	kernel_horizontal_flip = np.flip(kernel, axis=0)
	kernel_flipped = np.flip(kernel_horizontal_flip, axis=1)

	# To maintain same dtype as input image
	filtered_output = np.zeros((image.shape[0], image.shape[1]))
	# filtered_output = np.zeros_like(image)

	# Add zero padding to the image
	padded_image = np.zeros((
		image.shape[0] + kernel_flipped.shape[0] - 1,
		image.shape[1] + kernel_flipped.shape[1] -1
	))
	# Overlay the image onto the zero padding
	padded_image[kernel_flipped.shape[0]//2:-kernel_flipped.shape[0]//2+1, kernel_flipped.shape[1]//2:-kernel_flipped.shape[1]//2+1] = image

	# Convolution operation
	for x in range(image.shape[1]):
		for y in range(image.shape[0]):
			filtered_output[x,y] = (kernel_flipped * padded_image[x:x+kernel_flipped.shape[1], y:y+kernel_flipped.shape[0]]).sum()

	return filtered_output


def gaussian(x, mean, sigma, derivative_order):
	"""
	Compute 1D gaussian and its derivatives.

	:param x: Points at which to evaluate gaussian.
	:type x: numpy.ndarray
	:param mean: Mean of the guassian
	:type mean: float
	:param sigma: Standard deviation of the guassian.
	:type sigma: float
	:param derivative_order: Order of derivative.
	:type derivative_order: int
	:return: 1D gaussian or its derivative.
	:rtype: numpy.ndarray
	"""
	normalizing_factor = 1 / (np.sqrt(2 * np.pi * sigma**2))
	exponent = (np.exp(-0.5* ((x - mean)**2)/(sigma**2)))
	gaussian = normalizing_factor * exponent

	if derivative_order == 1:
		guassian_order1 = -gaussian * ((x - mean)/(sigma**2))
		return guassian_order1
	elif derivative_order == 2:
		guassian_order2 = gaussian * ((x - mean)**2 - (sigma**2))/(sigma**4)
		return guassian_order2
	return gaussian


def gaussian2d(size, sigma_x, sigma_y, ordx_derivative, ordy_derivative, orientation):

	half_size = (size - 1) // 2
	x, y = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
	points = np.vstack([x.ravel(), y.ravel()])

	c, s = np.cos(orientation), np.sin(orientation)
	rotation_matrix = np.array([[c, -s], [s, c]])
	rotated_points = rotation_matrix @ points
	x = rotated_points[0, :]
	y = rotated_points[1, :]

	gx = gaussian(x, 0, sigma_x, ordx_derivative)
	gy = gaussian(y, 0, sigma_y, ordy_derivative)

	gaussian_2d = np.reshape(gx * gy, (size, size))

	return gaussian_2d


def create_oriented_dog_filters(num_orientations, num_scales):
	"""
	Create a set of oriented DoG filters at different scales and orientations.

	:param num_orientations: Number of orientations.
	:type num_orientations: int
	:param num_scales: Number of scales.
	:type num_scales: int
	:return: List of DoG filters.
	:rtype: list of tuples of numpy.ndarray
	"""
	orientations = np.linspace(0, 360, num_orientations)
	scales = np.linspace(3, 7, num_scales)
	filters = []

	for scale in scales:
		for angle in orientations:
			gauss = gaussian2d(size=49, sigma_x=scale, sigma_y=scale, ordx_derivative=None, ordy_derivative=None, orientation=angle)
			# Sobel filters
			sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
			sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

			dog_x = convolve(gauss, sobel_x)
			dog_y = convolve(gauss, sobel_y)

			dog_filter = dog_x + dog_y
			filters.append(dog_filter)

	return filters

def plot_DOGBank(filters, plotname):
	"""
	Plot the DoG filter bank.

	:param filters: List of DoG filters.
	:type filters: list of tuples of numpy.ndarray
	"""
	num_filters = len(filters)
	fig, axes = plt.subplots(nrows=2, ncols=num_filters//2, figsize=(12, 3))
	for i, dog_filter in enumerate(filters):
		ax = axes[i//((num_filters)//2), i%((num_filters)//2)]
		ax.imshow(dog_filter, cmap='gray')
		ax.axis('off')

	plt.tight_layout()
	plt.savefig(plotname)
	plt.close()

####################################################################################################

##################################### Leung-Malik Filter Bank ######################################
def create_LM_filters(num_orientations, filter_size, version="LMS"):
	"""
	Generate Leung-Malik(LM) filter bank.

	:param version: Version of filter bank LMS for Small and LML for Large, defaults to "LMS"
	:type version: str, optional
	:raises ValueError: Version specified has to be either LMS or LML
	:return: List of LM filters.
	:rtype: list of numpy.ndarray
	"""
	# Scales for LMS and LML
	if version == "LMS":
		scales = [1, np.sqrt(2), 2, 2*np.sqrt(2)]  # Scales for LM Small
	elif version == "LML":
		scales = [np.sqrt(2), 2, 2*np.sqrt(2), 4]  # Scales for LM Large
	else:
		raise ValueError("Version must be either LMS or LML")

	orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)  # Orientations
	first_order_filters = len(scales[:-1]) * num_orientations
	second_order_filters = len(scales[:-1]) * num_orientations
	num_rotational_invariant_filters = 12

	total_filters = first_order_filters + second_order_filters + num_rotational_invariant_filters

	filter_bank = np.zeros((filter_size, filter_size, total_filters))
	filter_bank = [np.zeros((49, 49)) for _ in range(total_filters+1)]

	filter_count = 0
	for scale in scales[:3]:  # First three scales for derivative filters
		for angle in orientations:
			filter_bank[filter_count] = gaussian2d(filter_size, scale, 3 * scale, 1, 0, angle)
			filter_bank[filter_count + second_order_filters] = gaussian2d(filter_size, scale, 3 * scale, 2, 0, angle)
			filter_count += 1


	filter_count = first_order_filters + second_order_filters
	for i in range(len(scales)):
		laplacian_gaussian_x = gaussian2d(filter_size, scales[i], scales[i], 2, 0, 0)
		laplacian_gaussian_y = gaussian2d(filter_size, scales[i], scales[i], 0, 2, 0)
		filter_bank[filter_count] = laplacian_gaussian_x + laplacian_gaussian_y
		filter_count += 1

	for i in range(len(scales)):
		laplacian_gaussian_x = gaussian2d(filter_size, 3 * scales[i], 3 *scales[i], 2, 0, 0)
		laplacian_gaussian_y = gaussian2d(filter_size, 3 * scales[i], 3 *scales[i], 0, 2, 0)
		filter_bank[filter_count] = laplacian_gaussian_x + laplacian_gaussian_y
		filter_count += 1

	for i in range(len(scales)):
		filter_bank[filter_count] = gaussian2d(filter_size, scales[i], scales[i], 0, 0, 0)

		filter_count += 1

	return filter_bank


def plot_LMBank(filters, plotname):
	"""
	Plot the LM filter bank.

	:param filters: List of LM filters.
	:type filters: list of numpy.ndarray
	"""
	# Generate plot with 4 rows and 12 columns
	fig, axes = plt.subplots(nrows=4, ncols=12, figsize=(12, 8))

	for row in range(4):
		for col in range(12):
			if row < 3:
				if col < 6:
					index = row * 6 + col
				else:
					index = 18 + row * 6 + (col - 6)
			else:
				index = 36 + col
			ax = axes[row, col]
			ax.imshow(filters[index], cmap='gray')
			ax.axis('off')

	plt.tight_layout()
	plt.savefig(plotname)
	plt.close()

####################################################################################################

##################################### Gabor Filter Bank ######################################
import numpy as np
import matplotlib.pyplot as plt


def gabor(size, sigma, orientation, Lambda, psi, gamma):
	"""
	Gabor feature extraction.

	:param sigma: Standard deviation of the Gaussian envelope.
	:type sigma: float
	:param theta: Orientation of the normal to the parallel stripes of a Gabor function.
	:type theta: float
	:param Lambda: Wavelength of the sinusoidal factor.
	:type Lambda: float
	:param psi: Phase offset.
	:type psi: float
	:param gamma: Spatial aspect ratio.
	:type gamma: float
	:param nstd: Number of standard deviation sigma.
	:type nstd: int
	:return: Gabor filter.
	:rtype: numpy.ndarray
	"""
	sigma_x = sigma
	sigma_y = float(sigma) / gamma

	half_size = (size - 1) // 2
	x, y = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
	points = np.vstack([x.ravel(), y.ravel()])

	c, s = np.cos(orientation), np.sin(orientation)
	rotation_matrix = np.array([[c, -s], [s, c]])
	rotated_points = rotation_matrix @ points

	x_theta = rotated_points[0, :]
	y_theta = rotated_points[1, :]

	gabor_filter = np.exp(
		-0.5 * (x_theta**2 / sigma_x**2 + (y_theta**2 * gamma**2) / sigma_y**2)
	) * np.cos(2 * np.pi / Lambda * x_theta + psi)

	gabor_filter = np.reshape(gabor_filter, (size, size))

	return gabor_filter

def create_gabor_filters(size, sigmas, thetas, Lamda, psi, gamma):
	"""
	Create a set of Gabor filters at different orientations.

	:param sigma: Standard deviation of the Gaussian envelope.
	:type sigma: float
	:param nstds: Number of standard deviation sigma.
	:type nstds: int
	:param thetas: List of orientations.
	:type thetas: list of float
	:param Lamda: Wavelength of the sinusoidal factor.
	:type Lamda: float
	:param psi: Phase offset.
	:type psi: float
	:param gamma: Spatial aspect ratio.
	:type gamma: float
	:return: List of Gabor filters.
	:rtype: list of numpy.ndarray
	"""
	filters = []
	for sigma in (sigmas):
		for theta in (thetas):
			gb = gabor(size, sigma, theta, Lamda, psi, gamma)
			filters.append(gb)
	return filters

def plot_GaborBank(filters, plotname):
	"""
	Plot the Gabor filter bank.

	:param filters: List of Gabor filters.
	:type filters: list of numpy.ndarray
	"""
	nrows, ncols = 5, 8

	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 10))

	for i, filter in enumerate(filters):
		row = i // ncols
		col = i % ncols

		# Plot the filter
		ax = axes[row, col]
		ax.imshow(filter, cmap='gray')
		ax.axis('off')

	plt.tight_layout()
	plt.savefig(plotname)
	plt.close()
####################################################################################################

##################################### Half-disk masks ######################################
def HalfDiskFilterBank(radii, orientations):
    filter_bank = []
    orients = np.linspace(0, 360, orientations)

    for radius in radii:
        size = 2 * radius + 1
        for orient in orients:
            mask = np.zeros([size, size])
            for i in range(radius):
                for j in range(size):
                    dist = (i - radius)**2 + (j - radius)**2
                    if dist < radius**2:
                        mask[i, j] = 1

            half_mask = skimage.transform.rotate(mask, orient)
            half_mask = np.round(half_mask)

            half_mask_rot = skimage.transform.rotate(half_mask, 180, cval=1)
            half_mask_rot = np.round(half_mask_rot)

            filter_bank.append(half_mask)
            filter_bank.append(half_mask_rot)

    return filter_bank


def plot_save(filters, file_dir, cols):
	rows = np.ceil(len(filters)/cols).astype(int)
	plt.subplots(rows, cols, figsize=(15,15))
	for index in range(len(filters)):
		plt.subplot(rows, cols, index+1)
		plt.axis('off')
		plt.imshow(filters[index], cmap='gray')
	plt.savefig(file_dir)
	plt.close()

####################################################################################################

##################################### Texton Maps ######################################
def create_texton_map(filtered_image, K=64):
	"""
	Generate texton map using K-means clustering.

	:param filtered_image: Filtered image.
	:type filtered_image: numpy.ndarray
	:param K: _description_, defaults to 64
	:type K: int, optional
	:return: Texton map.
	:rtype: numpy.ndarray
	"""
	# Reshape the filter responses
	pixel_array = filtered_image.reshape(filtered_image.shape[0], -1).T

	# Cluster the filter responses using K-means
	kmeans = KMeans(n_clusters=K, random_state=0).fit(pixel_array)
	texton_ids = kmeans.labels_.reshape(filtered_image.shape[1:])

	return texton_ids
####################################################################################################

##################################### Brightness Map ######################################
def generate_brightness_map(image_path, K=16):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Flatten the image to a 2D array
    pixels = image.flatten().reshape(-1, 1)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=K, random_state=0).fit(pixels)
    brightness_ids = kmeans.labels_.reshape(image.shape)

    return brightness_ids

####################################################################################################

##################################### Color Map ######################################
def generate_color_map(image_path, K=16):
		# Load image
		image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Reshape the image to a 2D array
		pixels = image.reshape(-1, 3)

		# Apply K-means clustering
		kmeans = KMeans(n_clusters=K, random_state=0).fit(pixels)
		color_labels = kmeans.labels_.reshape(image.shape[:2])

		return color_labels
####################################################################################################

##################################### Texton Gradient ######################################
def gradient(image, bins, half_disk):
    """
    Compute texture gradients of the image using a bank of half-disk filters.

    :param image: Texton map of the image.
    :type image: numpy.ndarray
    :param bins: Number of texton bins.
    :type bins: int
    :param half_disk: Bank of half-disk filters.
    :type half_disk: list of numpy.ndarray
    :return: 3D matrix of texture gradient values.
    :rtype: numpy.ndarray
    """
    # Initialize a list to store the gradients
    gradients = []

    # Define the chi-square distance function
    def chi_square_dist(img, bin_id, filter1, filter2):
        tmp = np.float32(img == bin_id)
        g = cv2.filter2D(tmp, -1, filter1)
        h = cv2.filter2D(tmp, -1, filter2)
        chi_sqr = (g - h) ** 2 / (g + h + np.exp(-10))
        return chi_sqr / 2

    # Loop over each pair of half-disk filters
    for i in range(0, len(half_disk), 2):
        chi_sqr_dist = np.zeros_like(image, dtype=np.float32)
        for bin_id in range(bins):
            chi_sqr_dist += chi_square_dist(image, bin_id, half_disk[i], half_disk[i + 1])

        gradients.append(chi_sqr_dist)

    gradient_matrix = np.stack(gradients, axis=-1)

    return gradient_matrix
####################################################################################################
def main():

	IMAGES_PATH = "../BSDS500/Images"
	SOBEL_PATH = "../BSDS500/SobelBaseline"
	CANNY_PATH = "../BSDS500/CannyBaseline"

	TEXTON_MAPS_PATH = "../Outputs/TextonMaps"
	TEXTON_GRADIENTS_PATH = "../Outputs/TextonGradients"
	BRIGHTNESS_MAPS_PATH = "../Outputs/BrightnessMaps"
	BRIGHTNESS_GRADIENTS_PATH = "../Outputs/BrightnessGradients"
	COLOR_MAPS_PATH = "../Outputs/ColorMaps"
	COLOR_GRADIENTS_PATH = "../Outputs/ColorGradients"
	PBLITE_PATH = "../Outputs/PbliteOutputs"

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	# Generate 16 orientations and 2 scales of DoG filters
	dog_filters = create_oriented_dog_filters(num_orientations=16, num_scales=2)
	plot_DOGBank(dog_filters, '/home/megatron/workspace/WPI/Sem2/RBE549-Computer_Vision/submission/YourDirectoryID_hw0/Phase1/Code/filter_banks/DoG/DoG.png')

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	# Generate LM filters
	lms_filters = create_LM_filters(num_orientations=6, filter_size=49, version='LMS')  # Generate LM Small filters
	lml_filters = create_LM_filters(num_orientations=6, filter_size=49, version='LML')  # Generate LM Large filters
	lm_filters = lms_filters + lml_filters

	# Plot the LM filters
	plot_LMBank(lms_filters, '/home/megatron/workspace/WPI/Sem2/RBE549-Computer_Vision/submission/YourDirectoryID_hw0/Phase1/Code/filter_banks/LM/LM_Small.png')
	plot_LMBank(lml_filters, '/home/megatron/workspace/WPI/Sem2/RBE549-Computer_Vision/submission/YourDirectoryID_hw0/Phase1/Code/filter_banks/LM/LM_Large.png')

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	# Generate Gabor filters
	filter_size = 33
	num_orientations = 8
	sigmas = np.geomspace(3, 33, 5, endpoint=False)
	thetas = np.linspace(0, np.pi, num_orientations, endpoint=False)
	Lambda = 10
	psi = 0
	gamma = 1
	gabor_filters = create_gabor_filters(filter_size, sigmas, thetas, Lambda, psi, gamma)

	plot_GaborBank(gabor_filters, '/home/megatron/workspace/WPI/Sem2/RBE549-Computer_Vision/submission/YourDirectoryID_hw0/Phase1/Code/filter_banks/Gabor/Gabor.png')

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	half_disk = HalfDiskFilterBank([3, 9, 12], 8)
	plot_save(half_disk,"/home/megatron/workspace/WPI/Sem2/RBE549-Computer_Vision/submission/YourDirectoryID_hw0/Phase1/Code/filter_banks/Half_Disks/halfdisk.png",8)

	for image_name in tqdm(os.listdir(IMAGES_PATH)):
		image_path = os.path.join(IMAGES_PATH, image_name)
		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


		combined_filters = dog_filters + lm_filters + gabor_filters
		filtered_image = np.array([cv2.filter2D(image, -1, filter) for filter in combined_filters])

		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		texton_map = create_texton_map(filtered_image)
		if not os.path.exists(TEXTON_MAPS_PATH): os.makedirs(TEXTON_MAPS_PATH)
		filename = os.path.join(TEXTON_MAPS_PATH, 'texton_map_' + image_name)

		# Visualization
		plt.imsave(filename, texton_map, cmap='viridis')


		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		texton_gradient = gradient(texton_map, 64, half_disk)
		texton_gradient = np.mean(texton_gradient, axis=2)
		if not os.path.exists(TEXTON_GRADIENTS_PATH): os.makedirs(TEXTON_GRADIENTS_PATH)
		filename = os.path.join(TEXTON_GRADIENTS_PATH, 'texton_grad_' + image_name)

		# Visualization
		plt.imsave(filename, texton_gradient, cmap='viridis')

		"""
		Generate Brightness Map
		Perform brightness binning
		"""
		brightness_map = generate_brightness_map(image_path=image_path)
		if not os.path.exists(BRIGHTNESS_MAPS_PATH): os.makedirs(BRIGHTNESS_MAPS_PATH)
		filename = os.path.join(BRIGHTNESS_MAPS_PATH, 'brightness_map_' + image_name)

		# Visualization
		plt.imsave(filename, brightness_map, cmap='viridis')

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		brightness_gradient = gradient(brightness_map, 16, half_disk)
		brightness_gradient = np.mean(brightness_gradient, axis=2)
		if not os.path.exists(BRIGHTNESS_GRADIENTS_PATH): os.makedirs(BRIGHTNESS_GRADIENTS_PATH)
		filename = os.path.join(BRIGHTNESS_GRADIENTS_PATH, 'brightness_grad_' + image_name)

		# Visualization
		plt.imsave(filename, brightness_gradient, cmap='viridis')

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		color_map = generate_color_map(image_path=image_path)
		if not os.path.exists(COLOR_MAPS_PATH): os.makedirs(COLOR_MAPS_PATH)
		filename = os.path.join(COLOR_MAPS_PATH, 'color_map_' + image_name)

		# Visualization
		plt.imsave(filename, color_map, cmap='viridis')

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		color_gradient = gradient(color_map, 16, half_disk)
		color_gradient = np.mean(color_gradient, axis=2)
		if not os.path.exists(COLOR_GRADIENTS_PATH): os.makedirs(COLOR_GRADIENTS_PATH)
		filename = os.path.join(COLOR_GRADIENTS_PATH, 'color_grad_' + image_name)

		# Visualization
		plt.imsave(filename, color_gradient, cmap='viridis')

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		filename = os.path.join(SOBEL_PATH, image_name.split('.')[0] + '.png')
		sobel_baseline = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		filename = os.path.join(CANNY_PATH, image_name.split('.')[0] + '.png')
		canny_baseline = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		term1 = (texton_gradient + brightness_gradient + color_gradient) / 3
		term2 = (0.5 * sobel_baseline + 0.5 * canny_baseline)
		pb_lite = np.multiply(term1, term2)

		if not os.path.exists(PBLITE_PATH): os.makedirs(PBLITE_PATH)
		filename = os.path.join(PBLITE_PATH, 'pblite_' + image_name)
		plt.imsave(filename, pb_lite, cmap='gray')

if __name__ == '__main__':
    main()



