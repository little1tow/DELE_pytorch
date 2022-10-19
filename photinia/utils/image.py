# #!/usr/bin/env python3
#
# import random
#
# import cv2 as cv
# import numpy as np
# import scipy.ndimage as ndi
#
#
# def load_image(file_or_data, rgb=True):
#     if isinstance(file_or_data, str):
#         with open(file_or_data, 'rb') as f:
#             data = f.read()
#     else:
#         data = file_or_data
#     image = cv.imdecode(np.frombuffer(data, np.byte), cv.IMREAD_COLOR)
#     if rgb:
#         image = np.flip(image, axis=2)
#     return image
#
#
# def resize(image, height, width):
#     return cv.resize(image, (width, height))
#
#
# def pad(image, top, right=None, bottom=None, left=None):
#     h, w, c = image.shape
#     if right is None:
#         right = top
#     if bottom is None:
#         bottom = top
#     if left is None:
#         left = right
#     assert top >= 0 and right >= 0 and bottom >= 0 and left >= 0
#     image_new = np.zeros((top + h + bottom, left + w + right, c), image.dtype)
#     image_new[top:top + h, left:left + w, :] = image
#     return image_new
#
#
# def flip_lr(image):
#     return np.flip(image, axis=1)
#
#
# def random_flip_lr(image, prob=0.5):
#     return image if random.uniform(0.0, 1.0) < prob else flip_lr(image)
#
#
# def flip_ud(image):
#     return np.flip(image, axis=0)
#
#
# def random_flip_ud(image, prob=0.5):
#     return image if random.uniform(0.0, 1.0) < prob else flip_ud(image)
#
#
# def center_crop(image, height, width):
#     shape = image.shape
#     original_height = shape[0]
#     original_width = shape[1]
#     if original_height < height or original_width < width:
#         raise RuntimeError(
#             f'The input image ({original_height}, {original_width}) '
#             f'is smaller than ({height}, {width}).'
#         )
#     i = (original_height - height) // 2
#     j = (original_width - width) // 2
#     return image[i:i + height, j:j + width, :]
#
#
# def random_crop(image, height, width, padding=None):
#     if padding is not None:
#         image = pad(image, padding)
#     shape = image.shape
#     original_height = shape[0]
#     original_width = shape[1]
#     if original_height < height or original_width < width:
#         raise RuntimeError(
#             f'The input image ({original_height}, {original_width}) '
#             f'is smaller than ({height}, {width}).'
#         )
#     i = random.randint(0, original_height - height)
#     j = random.randint(0, original_width - width)
#     return image[i:i + height, j:j + width, :]
#
#
# def random_channel(image, intensity=0.1):
#     original_dtype = image.dtype
#     if original_dtype != np.float32:
#         image = np.array(image, np.float32)
#     image = np.rollaxis(image, 2, 0)
#     min_x, max_x = np.min(image), np.max(image)
#     intensity = intensity * (max_x - min_x)
#     channel_images = [
#         np.clip(channel + np.random.uniform(-intensity, intensity), min_x, max_x)
#         for channel in image
#     ]
#     image = np.stack(channel_images, axis=0)
#     image = np.rollaxis(image, 0, 2 + 1)
#     if image.dtype != original_dtype:
#         image = np.array(image, original_dtype)
#     return image
#
#
# def _trans_mat_offset_center(image, x, y):
#     o_x = float(x) / 2 + 0.5
#     o_y = float(y) / 2 + 0.5
#     offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
#     reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
#     transform_matrix = np.dot(np.dot(offset_matrix, image), reset_matrix)
#     return transform_matrix
#
#
# def _apply_transform(image,
#                      trans_mat,
#                      fill_mode='nearest',
#                      const_value=0.0):
#     """Apply the image transformation specified by a matrix.
#
#     Args:
#         image: The input image. (h,w, c)
#         trans_mat: The transform matrix.
#         fill_mode (str): one of `{'constant', 'nearest', 'reflect', 'wrap'}`
#         const_value (float): Value used for points outside the boundaries of the input if `mode='constant'`.
#
#     Returns:
#         The transformed version of the input.
#     """
#     image = np.rollaxis(image, 2, 0)
#     final_affine_matrix = trans_mat[:2, :2]
#     final_offset = trans_mat[:2, 2]
#     channel_images = [ndi.interpolation.affine_transform(
#         x_channel,
#         final_affine_matrix,
#         final_offset,
#         order=0,
#         mode=fill_mode,
#         cval=const_value) for x_channel in image]
#     image = np.stack(channel_images, axis=0)
#     image = np.rollaxis(image, 0, 2 + 1)
#     return image
#
#
# def random_rotate(image,
#                   degree,
#                   fill_mode='nearest',
#                   const_value=0.0):
#     """Performs a random rotation of a Numpy image tensor.
#
#     Args:
#         image: The input image. (h, w, c)
#         degree (float): The image will be rotated between -degree and degree.
#         fill_mode (str): one of `{'constant', 'nearest', 'reflect', 'wrap'}`
#         const_value (float): Value used for points outside the boundaries of the input if `mode='constant'`.
#
#     Returns:
#         The rotated image.
#     """
#     theta = np.pi / 180 * np.random.uniform(-degree, degree)
#     trans_mat = np.array(
#         [[np.cos(theta), -np.sin(theta), 0],
#          [np.sin(theta), np.cos(theta), 0],
#          [0, 0, 1]]
#     )
#     h, w = image.shape[0], image.shape[1]
#     trans_mat = _trans_mat_offset_center(trans_mat, h, w)
#     image = _apply_transform(image, trans_mat, fill_mode, const_value)
#     return image
