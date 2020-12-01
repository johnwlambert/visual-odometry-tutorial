
import copy
import pickle
import pdb
from typing import List, NamedTuple

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.se3 import SE3
from argoverse.utils.calibration import load_calib
from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t
from argoverse.data_loading.synchronization_database import get_timestamps_from_sensor_folder
from colour import Color
from pathlib import Path
from PIL import Image as PILImage
from PIL.ExifTags import GPSTAGS, TAGS
from scipy.spatial.transform import Rotation


from collect_ground_truth_corr import show_correspondence_lines
from utils.sensor_width_database import SensorWidthDatabase


"""
In the camera frame:

R: [-0.37137223 32.4745113  -0.42247361] as Euler
t: array([ 2.63715618, -0.0299745 , 12.05120079])

translation recovered only up to a scale
"""

class ImgKptCorrespondences(NamedTuple):
	X1: np.ndarray
	Y1: np.ndarray
	X2: np.ndarray
	Y2: np.ndarray


def load_pkl_correspondences(pkl_fpath: str):
	""" """
	with open(str(pkl_fpath), 'rb') as f:
		d = pickle.load(f)

	X1 = np.array(d['x1'])
	Y1 = np.array(d['y1'])
	X2 = np.array(d['x2'])
	Y2 = np.array(d['y2'])
	
	return ImgKptCorrespondences(X1,Y1,X2,Y2)


def get_exif(img_fpath: str):
	""" """
	original_image = PILImage.open(img_fpath)
	exif_data = original_image.getexif()
	if exif_data is not None:
		parsed_data = {}
		for tag, value in exif_data.items():
			if tag in TAGS:
				parsed_data[TAGS.get(tag)] = value
			elif tag in GPSTAGS:
				parsed_data[GPSTAGS.get(tag)] = value
			else:
				parsed_data[tag] = value

		exif_data = parsed_data
	return exif_data


def get_intrinsics_from_exif(exif_data, value_array):
	"""
	'Make', 'Model', 'LensModel'

	Example camera:
		Alpha a6000
		APS-C type (23.5 x 15.6 mm) 
		Max. image resolution:	6000 x 4000
	"""
	if exif_data is None or len(exif_data) == 0:
		return None

	focal_length_mm = exif_data.get('FocalLength')

	sensor_width_db = SensorWidthDatabase()
	sensor_width_mm = sensor_width_db.lookup(
		exif_data.get('Make'),
		exif_data.get('Model'),
	)

	img_h_px, img_w_px = value_array.shape[:2]
	focal_length_px = max(img_h_px, img_w_px) * focal_length_mm/sensor_width_mm

	center_x = img_w_px/2
	center_y = img_h_px/2

	K = np.array(
		[
			[focal_length_px, 0, center_x],
			[0,  focal_length_px, center_y],
			[0, 0, 1]
		])
	return K


def estimate_orb_correspondences():



	# (Pdb) p img1_exif_data[ 'ExifImageWidth']
	# 6000
	# (Pdb) p img1_exif_data[ 'ExifImageHeight']
	# 4000
	# (Pdb) p img1_exif_data[ 'FocalLength']
	# 16.0

	ESTIMATE_FROM_SCRATCH = False

	if ESTIMATE_FROM_SCRATCH:
		E, F, corr_data = ORB_estimate_pose(img1_fpath, img2_fpath, K1=K, K2=K)
		corr_img = show_correspondence_lines(img1, img2, corr_data.X1, corr_data.Y1, corr_data.X2, corr_data.Y2)
		plt.imshow(corr_img)
		plt.show()
		
		pts_left = np.hstack([corr_data.X1.reshape(-1,1), corr_data.Y1.reshape(-1,1) ]).astype(np.int32)
		pts_right = np.hstack([corr_data.X2.reshape(-1,1), corr_data.Y2.reshape(-1,1)]).astype(np.int32)

		if F is not None:
			draw_epilines(
				copy.deepcopy(pts_left),
				copy.deepcopy(pts_right),
				copy.deepcopy(img1),
				copy.deepcopy(img2),
				copy.deepcopy(F)
			)
			plt.show()

		if E is not None:
			F = get_fmat_from_emat(E, K1=K, K2=K)
			draw_epilines(
				copy.deepcopy(pts_left),
				copy.deepcopy(pts_right),
				copy.deepcopy(img1),
				copy.deepcopy(img2),
				copy.deepcopy(F)
			)
			plt.show()




	#ransac_model, ransac_inliers = cv2.findFundamentalMat(pts1, pts2, ransacReprojThreshold=opt.threshold, confidence=0.999)

	# pts1_virt, pts2_virt = compute_virtual_points(img_h, img_w, i2_F_i1)

	# corr_data = ImgKptCorrespondences(
	# 	X1=pts1_virt[:,0],
	# 	Y1=pts1_virt[:,1],
	# 	X2=pts2_virt[:,0],
	# 	Y2=pts2_virt[:,1]
	# )
	# corr_img = show_correspondence_lines(img1, img2, corr_data.X1, corr_data.Y1, corr_data.X2, corr_data.Y2)
	# plt.imshow(corr_img)
	plt.show()


def cartesian_to_homogeneous(pts):
	""" """
	n = pts.shape[0]
	return np.hstack([ pts, np.ones((n,1)) ])


def compute_virtual_points(img_h: int, img_w: int, i2_F_i1: np.ndarray, step: float = 0.5):
	"""Compute virtual points for a single sample.
	Args:
		index (int): sample index
	Returns:
		tuple: virtual points in first image, virtual points in second image
	"""
	
	# set grid points for each image
	grid_x, grid_y = np.meshgrid(
		np.arange(0, img_w, int(step * img_w)),
		np.arange(0, img_h, int(step * img_h))
	)
	num_points_eval = len(grid_x.flatten())

	grid_x = grid_x.flatten().reshape(-1,1)
	grid_y = grid_y.flatten().reshape(-1,1)

	pts1_grid = np.hstack([grid_x, grid_y]).reshape(1,-1,2)
	pts2_grid = copy.deepcopy(pts1_grid).reshape(1,-1,2)

	pts1_virt, pts2_virt = cv2.correctMatches(i2_F_i1, pts1_grid, pts2_grid)
	pts1_virt = pts1_virt.squeeze()
	pts2_virt = pts2_virt.squeeze()

	valid = np.logical_and.reduce(
		[
			np.logical_not(np.isnan(pts1_virt[:, 0])),
			np.logical_not(np.isnan(pts1_virt[:, 1])),
			np.logical_not(np.isnan(pts2_virt[:, 0])),
			np.logical_not(np.isnan(pts2_virt[:, 1]))
		])
	valid_idx = np.where(valid)[0]
	good_pts = len(valid_idx)

	pts1_virt = pts1_virt[valid_idx]
	pts2_virt = pts2_virt[valid_idx]

	return pts1_virt, pts2_virt


def compute_essential_matrix(i2_R_i1: np.ndarray, i2_t_i1: np.ndarray) -> np.ndarray:
	"""Compute essential matrix
	Args: 
		i2_R_i1
		i2_t_i1

	Returns:
		i2_E_i1: essential matrix
	"""
	tx = skew_symmetric(i2_t_i1)
	i2_E_i1 = tx @ i2_R_i1
	return i2_E_i1


def skew_symmetric(t: np.ndarray) -> np.ndarray:
	"""Compute skew symmetric matrix of vector t
	Args:
		t: vector of shape (3,)
	Returns:
		M: skew-symmetric matrix of shape (3, 3)
	"""
	M = np.array(
		[ [0, -t[2], t[1] ],
		[ t[2], 0, -t[0] ],
		[ -t[1], t[0], 0]
	])
	return M


def get_fmat_from_emat(i2_E_i1: np.ndarray, K1: np.ndarray, K2: np.ndarray):
	""" """
	i2_F_i1 = np.linalg.inv(K2).T @ i2_E_i1 @ np.linalg.inv(K1)
	return i2_F_i1



def draw_epipolar_lines(F, img_left, img_right, pts_left, pts_right):
    """
    Draw the epipolar lines given the fundamental matrix, left right images
    and left right datapoints

    You do not need to modify anything in this function, although you can if
    you want to.
    :param F: 3 x 3; fundamental matrix
    :param img_left:
    :param img_right:
    :param pts_left: N x 2
    :param pts_right: N x 2
    :return:
    """
    # lines in the RIGHT image
    # corner points
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([img_right.shape[1], 0, 1])
    p_bl = np.asarray([0, img_right.shape[0], 1])
    p_br = np.asarray([img_right.shape[1], img_right.shape[0], 1])

    # left and right border lines
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    fig, ax = plt.subplots()
    ax.imshow(img_right)
    ax.autoscale(False)
    ax.scatter(pts_right[:, 0], pts_right[:, 1], marker='o', s=20, c='yellow',
        edgecolors='red')
    for p in pts_left:
        p = np.hstack((p, 1))[:, np.newaxis]
        l_e = np.dot(F, p).squeeze()  # epipolar line
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        x = [p_l[0]/p_l[2], p_r[0]/p_r[2]]
        y = [p_l[1]/p_l[2], p_r[1]/p_r[2]]
        ax.plot(x, y, linewidth=1, c='blue')

    # lines in the LEFT image
    # corner points
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([img_left.shape[1], 0, 1])
    p_bl = np.asarray([0, img_left.shape[0], 1])
    p_br = np.asarray([img_left.shape[1], img_left.shape[0], 1])

    # left and right border lines
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    fig, ax = plt.subplots()
    ax.imshow(img_left)
    ax.autoscale(False)
    ax.scatter(pts_left[:, 0], pts_left[:, 1], marker='o', s=20, c='yellow',
        edgecolors='red')
    for p in pts_right:
        p = np.hstack((p, 1))[:, np.newaxis]
        l_e = np.dot(F.T, p).squeeze()  # epipolar line
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        x = [p_l[0]/p_l[2], p_r[0]/p_r[2]]
        y = [p_l[1]/p_l[2], p_r[1]/p_r[2]]
        ax.plot(x, y, linewidth=1, c='blue')




def egomotion_unit_test():
	"""
	1R2 bring points in 2's frame into 1's frame
	1R2 is the relative rotation from 1's frame to 2's frame
	"""
	city_SE3_egot1 = SE3(rotation=np.eye(3), translation=np.array([3040,0,0]))
	city_SE3_egot2 = SE3(rotation=np.eye(3), translation=np.array([3050,0,0]))

	egot1_SE3_city = city_SE3_egot1.inverse()
	egot1_SE3_egot2 = egot1_SE3_city.right_multiply_with_se3(city_SE3_egot2)

	assert np.allclose(egot1_SE3_egot2.translation, np.array([10,0,0]))



colors = np.random.rand(50,3)
num_colors = colors.shape[0]

def drawlines(img1: np.ndarray, img2: np.ndarray, lines, pts1, pts2):
	''' img1 - image on which we draw the epilines for the points in img2
	lines - corresponding epilines '''
	r,c, _ = img1.shape

	for i, (r,pt1,pt2) in enumerate(zip(lines,pts1,pts2)):
		color = colors[i % num_colors]
		color = tuple(color.tolist())
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		img1 = cv2.line(img1, (x0,y0), (x1,y1), color,10)
		img1 = cv2.circle(img1,tuple(pt1),20,color,-1)
		img2 = cv2.circle(img2,tuple(pt2),20,color,-1)
	return img1,img2



def draw_epilines(pts1: np.ndarray, pts2: np.ndarray, img1: np.ndarray, img2: np.ndarray, F: np.ndarray) -> np.ndarray:
	# Find epilines corresponding to points in right image (second image) and
	# drawing its lines on left image
	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
	# Find epilines corresponding to points in left image (first image) and
	# drawing its lines on right image
	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
	plt.subplot(121),plt.imshow(img5)
	plt.subplot(122),plt.imshow(img3)
	plt.show()



def ORB_estimate_pose(img1_fpath, img2_fpath, K1, K2, ratio = 0.9, fmat: bool = False):
	""" """
	detector = cv2.ORB_create()

	img1 = cv2.imread(img1_fpath)
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

	img2 = cv2.imread(img2_fpath)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	kp1, desc1 = detector.detectAndCompute(img1, None)
	kp2, desc2 = detector.detectAndCompute(img2, None)

	# feature matching
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1, desc2, k=2)

	good_matches = []
	pts1 = []
	pts2 = []

	#side information for the network (matching ratios in this case)
	ratios = []

	
	print("")
	if ratio < 1.0:
		print("Using Lowe's ratio filter with", ratio)

	for (m,n) in matches:
		if m.distance < ratio*n.distance: # apply Lowe's ratio filter
			good_matches.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)
			ratios.append(m.distance / n.distance)

	print("Number of valid matches:", len(good_matches))

	pts1 = np.array([pts1])
	pts2 = np.array([pts2])

	ratios = np.array([ratios])
	ratios = np.expand_dims(ratios, 2)

	threshold = 0.001
	# ------------------------------------------------
	# fit fundamental or essential matrix using OPENCV
	# ------------------------------------------------
	if fmat:

		# === CASE FUNDAMENTAL MATRIX =========================================

		F, ransac_inliers = cv2.findFundamentalMat(pts1, pts2, ransacReprojThreshold=threshold, confidence=0.99999)
		E = None
	else:
		# === CASE ESSENTIAL MATRIX =========================================

		# normalize key point coordinates when fitting the essential matrix
		undist_pts1 = cv2.undistortPoints(pts1, K1, None)
		undist_pts2 = cv2.undistortPoints(pts2, K2, None)

		K = np.eye(3)
		E, ransac_inliers = cv2.findEssentialMat(undist_pts1, undist_pts2, K, method=cv2.RANSAC, prob=0.999, threshold=threshold)
		F = None

	print("\n=== Model found by RANSAC: ==========\n")
	print(E)
	print(F)

	print("\nRANSAC Inliers:", ransac_inliers.sum())

	
	kp1 = pts1.squeeze()[ransac_inliers.squeeze() ==1 ]
	kp2 = pts2.squeeze()[ransac_inliers.squeeze() ==1 ]

	X1 = kp1[:,0]
	Y1 = kp1[:,1]

	X2 = kp2[:,0]
	Y2 = kp2[:,1]

	corr_data = ImgKptCorrespondences(X1,Y1,X2,Y2)
	return E, F, corr_data

def array_of_keypoints(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
	""" """
	feat_list = [[kp.pt[0], kp.pt[1]] for kp in keypoints]
	return np.array(feat_list, dtype=np.float32)


	# match_img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2, matchColor=(75,180,60), matchesMask = ransac_inliers)
	# plt.imshow(match_img_ransac)
	# plt.show()



def plot_argoverse_trajectory(ts1: int, ts2: int, dataset_dir: str, log_id: str):
	""" """
	city_SE3_egot1 = get_city_SE3_egovehicle_at_sensor_t(ts1, dataset_dir, log_id) 
	city_SE3_egot2 = get_city_SE3_egovehicle_at_sensor_t(ts2, dataset_dir, log_id) 

	# plot the vehicle movement
	sensor_folder_wildcard = f"{dataset_dir}/{log_id}/poses/city_SE3_egovehicle*.json"
	lidar_timestamps = get_timestamps_from_sensor_folder(sensor_folder_wildcard)

	colors_arr = np.array(
		[ [color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), len(lidar_timestamps) // 49 ) ]
	).squeeze()

	plt.close('all')
	for k,ts in enumerate(lidar_timestamps[::50]):
		city_SE3_egov_t = get_city_SE3_egovehicle_at_sensor_t(ts, dataset_dir, log_id) 
		t = city_SE3_egov_t.translation
		plt.scatter(t[0], t[1], 20, marker='.', color=colors_arr[k])

	t1 = city_SE3_egot1.translation
	t2 = city_SE3_egot2.translation

	print('t1 =', t1)
	print('t2 =', t2)

	plt.scatter(t1[0], t1[1], 10, marker='o', color='m')
	plt.scatter(t2[0], t2[1], 10, marker='o', color='c')

	posx_1 = city_SE3_egot1.transform_point_cloud(np.array([[3,0,0]])).squeeze()
	posy_1 = city_SE3_egot1.transform_point_cloud(np.array([[0,3,0]])).squeeze()

	posx_2 = city_SE3_egot2.transform_point_cloud(np.array([[3,0,0]])).squeeze()
	posy_2 = city_SE3_egot2.transform_point_cloud(np.array([[0,3,0]])).squeeze()

	plt.plot([t1[0], posx_1[0]], [t1[1], posx_1[1]], 'b')
	plt.plot([t1[0], posy_1[0]], [t1[1], posy_1[1]], 'k')

	plt.plot([t2[0], posx_2[0]], [t2[1], posx_2[1]], 'b')
	plt.plot([t2[0], posy_2[0]], [t2[1], posy_2[1]], 'k')

	plt.axis('equal')
	plt.title('Egovehicle trajectory')
	plt.xlabel('x city coordinate')
	plt.ylabel('y city coordinate')
	plt.show()

	egot1_SE3_city = city_SE3_egot1.inverse()
	egot1_SE3_egot2 = egot1_SE3_city.right_multiply_with_se3(city_SE3_egot2)

	# rotates i1's frame to i2's frame
	# 1R2 bring points in 2's frame into 1's frame
	# 1R2 is the relative rotation from 1's frame to 2's frame
	egot1_R_egot2 = egot1_SE3_egot2.rotation
	egot1_t_egot2 = egot1_SE3_egot2.translation

	r = Rotation.from_matrix(egot1_R_egot2)
	print('egot1_R_egot2 from ground truth: ', r.as_euler('zyx', degrees=True))
	print('egot1_t_egot2 = ', np.round(egot1_t_egot2,2))


def plot_argoverse_epilines_from_ground_truth_poses(ts1: int, ts2: int, img1, img2, K):
	""" """
	city_SE3_egot1 = get_city_SE3_egovehicle_at_sensor_t(ts1, dataset_dir, log_id) 
	city_SE3_egot2 = get_city_SE3_egovehicle_at_sensor_t(ts2, dataset_dir, log_id) 

	camera_T_egovehicle = calib_dict['ring_front_center'].extrinsic
	camera_T_egovehicle = SE3(rotation=camera_T_egovehicle[:3,:3], translation=camera_T_egovehicle[:3,3])
	egovehicle_T_camera = camera_T_egovehicle.inverse()

	city_SE3_cam1 = city_SE3_egot1.compose(egovehicle_T_camera)
	city_SE3_cam2 = city_SE3_egot2.compose(egovehicle_T_camera)

	cam1_SE3_city = city_SE3_cam1.inverse()
	cam1_SE3_cam2 = cam1_SE3_city.compose(city_SE3_cam2)

	# rotates i1's frame to i2's frame
	# 1R2 bring points in 2's frame into 1's frame
	# 1R2 is the relative rotation from 1's frame to 2's frame
	cam1_R_cam2 = cam1_SE3_cam2.rotation
	cam1_t_cam2 = cam1_SE3_cam2.translation

	r = Rotation.from_matrix(cam1_R_cam2)
	print('cam1_R_cam2 =', r.as_euler('zyx', degrees=True))
	print('Recover t=', cam1_t_cam2, ' up to scale ', cam1_t_cam2 / np.linalg.norm(cam1_t_cam2))

	cam2_SE3_cam1 = cam1_SE3_cam2.inverse()
	cam2_R_cam1 = cam2_SE3_cam1.rotation
	cam2_t_cam1 = cam2_SE3_cam1.translation

	# use correct ground truth relationship to generate gt_E
	# and then generate correspondences using
	cam2_E_cam1 = compute_essential_matrix(cam2_R_cam1, cam2_t_cam1)
	cam2_F_cam1 = get_fmat_from_emat(cam2_E_cam1, K1=K, K2=K)

	img_h, img_w, _ = img1.shape

	pkl_fpath = f'/Users/johnlambert/Downloads/visual-odometry-tutorial/labeled_correspondences/argoverse_1_E_0.pkl'
	corr_data = load_pkl_correspondences(pkl_fpath)

	pts_left = np.hstack([corr_data.X1.reshape(-1,1), corr_data.Y1.reshape(-1,1) ]).astype(np.int32)
	pts_right = np.hstack([corr_data.X2.reshape(-1,1), corr_data.Y2.reshape(-1,1)]).astype(np.int32)

	pdb.set_trace()
	draw_epilines(pts_left, pts_right, img1, img2, cam2_F_cam1)
	plt.show()
	draw_epipolar_lines(cam2_F_cam1, img1, img2, pts_left, pts_right)

	pts_left = cartesian_to_homogeneous(pts_left)
	pts_right = cartesian_to_homogeneous(pts_right)

	for (pt1, pt2) in zip(pts_left, pts_right):
		epi_error = pt2.dot(cam2_F_cam1).dot(pt1)
		print('Error: ', epi_error)


def plot_argoverse_epilines_from_annotated_correspondences(img1: np.ndarray, img2: np.ndarray, K: np.ndarray):
	""" """
	pkl_fpath = f'/Users/johnlambert/Downloads/visual-odometry-tutorial/labeled_correspondences/argoverse_1_E_0.pkl'
	corr_data = load_pkl_correspondences(pkl_fpath)

	corr_img = show_correspondence_lines(img1, img2, corr_data.X1, corr_data.Y1, corr_data.X2, corr_data.Y2)
	plt.imshow(corr_img)
	plt.show()

	img1_kpts = np.hstack([ corr_data.X1.reshape(-1,1), corr_data.Y1.reshape(-1,1) ]).astype(np.int32)
	img2_kpts = np.hstack([ corr_data.X2.reshape(-1,1), corr_data.Y2.reshape(-1,1) ]).astype(np.int32)

	cam2_E_cam1, inlier_mask = cv2.findEssentialMat(img1_kpts, img2_kpts, K, method=cv2.RANSAC, threshold=0.1)
	
	print('Num inliers: ', inlier_mask.sum())
	cam2_F_cam1 = get_fmat_from_emat(cam2_E_cam1, K1=K, K2=K)
	_num_inlier, cam2_R_cam1, cam2_t_cam1, _ = cv2.recoverPose(cam2_E_cam1, img1_kpts, img2_kpts, mask=inlier_mask)

	r = Rotation.from_matrix(cam2_R_cam1)
	print('cam2_R_cam1 recovered from correspondences', r.as_euler('zyx', degrees=True))
	print('cam2_t_cam1: ', np.round(cam2_t_cam1.squeeze(), 2))

	cam2_SE3_cam1 = SE3(cam2_R_cam1, cam2_t_cam1.squeeze() )
	cam1_SE3_cam2 = cam2_SE3_cam1.inverse()
	cam1_R_cam2 = cam1_SE3_cam2.rotation
	cam1_t_cam2 = cam1_SE3_cam2.translation

	r = Rotation.from_matrix(cam1_R_cam2)
	print('cam1_R_cam2: ', r.as_euler('zyx', degrees=True)) ## prints "[-0.32  33.11 -0.45]"
	print('cam1_t_cam2: ', np.round(cam1_t_cam2,2)) ## [0.21 0.   0.98]

	pdb.set_trace()
	draw_epilines(img1_kpts, img2_kpts, img1, img2, cam2_F_cam1)
	plt.show()

	draw_epipolar_lines(cam2_F_cam1, img1, img2, img1_kpts, img2_kpts)
	plt.show()


def plot_backyard_epilines():
	""" """

	# i = 1
	# j = 2

	i = 0
	j = 1

	img1_fpath = f'{img_dir}/{img_names[i]}'
	img2_fpath = f'{img_dir}/{img_names[j]}'

	img1_exif_data = get_exif(img1_fpath)
	img2_exif_data = get_exif(img2_fpath)

	img1_exif_data['Model'] = "Sony Alpha a6000" # "Alpha a6000"
	K = get_intrinsics_from_exif(img1_exif_data, img1)

	img_dir = '/Users/johnlambert/Downloads/homography_scenes'
	planar_img_names = [
		'plane_img1.jpg',
		'plane_img2.jpg',
		'plane_img3.jpg'
	]
	

	img_names = [
		'outdoor_img0.JPG',
		'outdoor_img1.JPG',
		'outdoor_img2.JPG'
	]
	img_dir = '/Users/johnlambert/Downloads/GTSFM/outdoor_imgs'
	for img_name in img_names:
		pass



if __name__ == '__main__':

	log_id = '273c1883-673a-36bf-b124-88311b1a80be'
	dataset_dir = '/Users/johnlambert/Downloads/visual-odometry-tutorial/train1'

	# img_names = [
	# 	'ring_front_center_315975640448534784.jpg',
	# 	'ring_front_center_315975643412234000.jpg'
	# ]
	img_dir = '/Users/johnlambert/Downloads/visual-odometry-tutorial/train1/273c1883-673a-36bf-b124-88311b1a80be/ring_front_center'
	dataset_name = 'argoverse'

	ts1 = 315975640448534784 # nano-second timestamp
	ts2 = 315975643412234000

	img1_fpath = f'{img_dir}/ring_front_center_{ts1}.jpg'
	img2_fpath = f'{img_dir}/ring_front_center_{ts2}.jpg'

	img1 = imageio.imread(img1_fpath).astype(np.float32) / 255
	img2 = imageio.imread(img2_fpath).astype(np.float32) / 255
	# plt.imshow(img)
	# plt.show()

	if dataset_name == 'argoverse':
		calib_fpath = '/Users/johnlambert/Downloads/visual-odometry-tutorial/train1/273c1883-673a-36bf-b124-88311b1a80be/vehicle_calibration_info.json'
		calib_dict = load_calib(calib_fpath)
		K = calib_dict['ring_front_center'].K[:3,:3]

		# get the ground truth pose
		# timestamps
		# ts1 = Path(img1_fpath).stem.split('_')[-1]
		# ts2 = Path(img2_fpath).stem.split('_')[-1]


	#plot_argoverse_trajectory(ts1, ts2, dataset_dir, log_id)
	plot_argoverse_epilines_from_ground_truth_poses(ts1, ts2, img1, img2, K)
	#plot_argoverse_epilines_from_annotated_correspondences(img1, img2, K)

	# plot_backyard_epilines()


	# estimate_orb_correspondences()
	#egomotion_unit_test()


