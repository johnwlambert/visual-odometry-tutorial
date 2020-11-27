#!/usr/bin/python3

# Provides an interactive method to specify and then save many point correspondences
# between two photographs, which will be used to generate a projective
# transformation.

# Pick a dozen corresponding points throughout the images, although more is
# better.

import argparse
from pathlib import Path
import pdb
import pickle

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


FIGURE_HEIGHT = 6
FIGURE_WIDTH = 10
plt.rcParams["figure.figsize"] = (FIGURE_WIDTH,FIGURE_HEIGHT)


def show_correspondence_lines(imgA, imgB, X1, Y1, X2, Y2, line_colors=None):
	"""
	Visualizes corresponding points between two images by drawing a line segment
	between the two images for each (x1,y1) (x2,y2) pair.

	Args:
	- imgA: A numpy array of shape (M,N,3)
	- imgB: A numpy array of shape (D,E,3)
	- x1: A numpy array of shape (k,) containing x-locations of keypoints in imgA
	- y1: A numpy array of shape (k,) containing y-locations of keypoints in imgA
	- x2: A numpy array of shape (j,) containing x-locations of keypoints in imgB
	- y2: A numpy array of shape (j,) containing y-locations of keypoints in imgB
	- line_colors: A numpy array of shape (N x 3) with colors of correspondence lines (optional)

	Returns:
	- newImg: A numpy array of shape (max(M,D), N+E, 3)
	"""
	newImg = hstack_images(imgA, imgB)
	shiftX = imgA.shape[1]
	X1 = X1.astype(np.int)
	Y1 = Y1.astype(np.int)
	X2 = X2.astype(np.int)
	Y2 = Y2.astype(np.int)

	dot_colors = np.random.rand(len(X1), 3)
	if line_colors is None:
		line_colors = dot_colors

	for x1, y1, x2, y2, dot_color, line_color in zip(X1, Y1, X2, Y2, dot_colors,line_colors):
		newImg = cv2.circle(newImg, (x1, y1), 5, dot_color, -1)
		newImg = cv2.circle(newImg, (x2+shiftX, y2), 5, dot_color, -1)
		newImg = cv2.line(newImg, (x1, y1), (x2+shiftX, y2), line_color, 5, cv2.LINE_AA)
	return newImg


def hstack_images(imgA, imgB):
    """
    Stacks 2 images side-by-side and creates one combined image.

    Args:
    - imgA: A numpy array of shape (M,N,3) representing rgb image
    - imgB: A numpy array of shape (D,E,3) representing rgb image

    Returns:
    - newImg: A numpy array of shape (max(M,D), N+E, 3)
    """
    Height = max(imgA.shape[0], imgB.shape[0])
    Width  = imgA.shape[1] + imgB.shape[1]

    newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
    newImg[:imgA.shape[0], :imgA.shape[1], :] = imgA
    newImg[:imgB.shape[0], imgA.shape[1]:, :] = imgB

    return newImg


class CorrespondenceAnnotator(object):
	def __init__(self, img_fpath1: str, img_fpath2: str, experiment_name: str):

		self.image1 = imageio.imread(img_fpath1)
		self.image2 = imageio.imread(img_fpath2)
		self.corr_file = Path(f'{experiment_name}.pkl')
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))
		self.ax1 = ax1
		self.ax2 = ax2
		self.x1 = [] # x locations in image 1
		self.y1 = [] # y locations in image 1
		self.x2 = [] # corresponding x locations in image 2
		self.y2 = [] # corresponding y locations in image 2

	def collect_ground_truth_corr(self):
		"""
		Collect ground truth image-to-image correspondences by manually annotating them.

		This function checks if some corresponding points are already saved, and
		if so, resumes work from there.
		"""
		if self.corr_file.exists():
			self.load_pkl_correspondences()

			# The correspondences that already exist
			corr_image = show_correspondence_lines(	self.image1, self.image2, 
													np.array(self.x1), np.array(self.y1), 
													np.array(self.x2), np.array(self.y2))
		else:
			self.x1 = [] 
			self.y1 = [] 
			self.x2 = [] 
			self.y2 = [] 

		self.ax1.imshow(self.image1)
		self.ax2.imshow(self.image2)

		self.ax1.axis('off')
		self.ax2.axis('off')

		self.mark_corrs_with_clicks()
		self.dump_pkl_correspondences()

		corr_image = show_correspondence_lines(	self.image1, self.image2, 
												np.array(self.x1), np.array(self.y1), 
												np.array(self.x2), np.array(self.y2))
		plt.gcf().clear()
		plt.imshow(corr_image)
		plt.show()

	def load_pkl_correspondences(self):
		with open(str(self.corr_file), 'rb') as f:
			d = pickle.load(f)

		self.x1 = d['x1']
		self.y1 = d['y1']
		self.x2 = d['x2']
		self.y2 = d['y2']

	def dump_pkl_correspondences(self):
		print('saving matched points')
		data_dict = {}
		data_dict['x1'] = self.x1
		data_dict['y1'] = self.y1
		data_dict['x2'] = self.x2
		data_dict['y2'] = self.y2

		with open(str(self.corr_file), 'wb') as f:
			pickle.dump(data_dict,f)

	def mark_corrs_with_clicks(self):
		"""
		Mark correspondences with clicks
		"""
		print('Exit the matplotlib window to stop annotation.')
		# title = 'Click on a point in the left window\n'
		# title += 'then on a point in the right window.\n'
		# title += 'Exit the matplotlib window to stop annotation.\n'
		# title += 'Afterwards, you will see the plotted correspondences.'
		#self.ax1.set_title(title)
		while(1):
			pt = plt.ginput(1)
			if len(pt) == 0:
				break
			x = pt[0][0]
			y = pt[0][1]

			self.ax1.scatter(x,y,30,color='r', marker='o')
			self.x1 += [x]
			self.y1 += [y]

			pt = plt.ginput(1)
			if len(pt) == 0:
				break
			x = pt[0][0]
			y = pt[0][1]

			self.ax2.scatter(x,y,30,color='r', marker='o')
			self.x2 += [x]
			self.y2 += [y]
		    
			print('({}, {}) matches to ({},{})'.format(	self.x1[-1], 
														self.y1[-1], 
														self.x2[-1], 
														self.y2[-1]))
			print('{} total points corresponded'.format(len(self.x1)))

if __name__ == '__main__':
	""" Parse command-line arguments and launch tool."""
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--img_fpath1",
		type=str,
		required=True,
		help=""
	)
	parser.add_argument(
		"--img_fpath2",
		type=str,
		required=True,
		help=""
	)
	parser.add_argument(
		"--experiment_name",
		type=str,
		required=True,
		help=""
	)
	args = parser.parse_args()

	ca = CorrespondenceAnnotator(args.img_fpath1, args.img_fpath2, args.experiment_name)
	ca.collect_ground_truth_corr()
