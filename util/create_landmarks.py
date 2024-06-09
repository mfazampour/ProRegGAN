import numpy as np
import SimpleITK as sitk
import torch

## for testing
from data import create_dataset
from options.train_options import TrainOptions


## for testing


def getLandmarks(filename, landmarks_filename):
    # Read the image
    image = sitk.ReadImage(filename)

    # Create an image of the same size as the original image
    binary_image = sitk.GetImageFromArray(np.zeros_like(sitk.GetArrayFromImage(image)))
    binary_image.CopyInformation(image)

    # Define the radius of the sphere in voxels
    radius = 2

    # Read the landmarks
    landmarks = open(landmarks_filename, "r")
    lm = landmarks.readlines()
    landmarks.close()
    world_landmark = []
    for x in lm:
        indexes = list(map(float, x.split()))
        world_landmark.append(indexes)

    # Iterate over each landmark
    for i in range(len(world_landmark)):
        # Convert the physical point to the voxel index
        landmark_voxel_coords = image.TransformPhysicalPointToIndex(world_landmark[i])
        # Set the binary image at the index to have the value 1
        # Iterate over each voxel in the binary image
        for x in range(landmark_voxel_coords[0] - radius, landmark_voxel_coords[0] + radius + 1):
            for y in range(landmark_voxel_coords[1] - radius, landmark_voxel_coords[1] + radius + 1):
                for z in range(landmark_voxel_coords[2] - radius, landmark_voxel_coords[2] + radius + 1):
                    # check if x,y,z are within the image
                    if x < 0 or x >= binary_image.GetWidth() or y < 0 or y >= binary_image.GetHeight() or z < 0 or z >= binary_image.GetDepth():
                        continue

                    # Check if the voxel is within the sphere
                    if np.linalg.norm(np.array([x, y, z]) - np.array(landmark_voxel_coords)) <= radius:
                        binary_image.SetPixel(x, y, z, i + 1)

    return torch.FloatTensor(sitk.GetArrayFromImage(binary_image).astype(np.int32)).transpose(0, 2)


#
def getLandmarks_old(filename, landmarks_filename):
    reader = sitk.ImageFileReader()

    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    landmarks = open(landmarks_filename, "r")
    lm = landmarks.readlines()
    landmarks.close()
    world_landmark = []
    for x in lm:
        indexes = list(map(float, x.split()))
        indexes = np.append(indexes, [1])
        world_landmark.append(indexes)

    file = open(filename, "r")
    lines = file.readlines()
    file.close()

    metdata = []
    for x in lines:
        if x.split()[0] == "Position" or x.split()[0] == "Orientation":
            metdata.append(x.replace('=', '').split()[1:])

    orientation = np.reshape([float(i) for i in metdata[1]], (3, 3))
    position = np.reshape([float(i) for i in metdata[0]], (3, 1))

    rotation_mtrx = np.concatenate((orientation, [[0, 0, 0]]), axis=0)
    rotation_mtrx = np.concatenate((rotation_mtrx, [[0], [0], [0], [1]]), axis=1)

    I = np.identity(4)
    C = np.multiply(position, -1)
    translation_matrix = I
    translation_matrix[0:3, 3] = np.reshape(C, (3))

    new_landmark_pos = []
    for i in range(len(world_landmark)):
        pos = translation_matrix.dot(np.reshape(world_landmark[i], (4, 1)))
        pos = rotation_mtrx.dot(pos)
        pos = correct_out_of_bounds(pos[0:3])
        new_landmark_pos.append(pos)

    size = reader.GetSize()
    # print(f"size from reader {size}", flush = True)
    #
    #  print(size,flush = True)
    A = 1
    image_3D = sitk.Image(size[0], size[1], size[2], sitk.sitkInt16)
    for i in range(len(new_landmark_pos)):
        x = int(new_landmark_pos[i][0])
        y = int(new_landmark_pos[i][1])
        z = int(new_landmark_pos[i][2])

        # print(str(x)+ ' ' +str(y)+ ' '+str(z), flush=True)
        try:
            image_3D.SetPixel(x, y, z, i + 1)
        except:
            #  print(str(size) + str(x)+ ' ' +str(y)+ ' '+str(z))
            print(f'patient is {filename} size is {image_3D.GetSize()} and location is {x}, {y}, {z}')
            print(f' pixel value {image_3D.GetPixel(x, y, z, )}')

        for index in range(1, 6):
            #  image_3D.SetPixel(x, y, z, i+1)
            try:
                image_3D.SetPixel(x + index, y, z, (i + 1))
                image_3D.SetPixel(x + index, y + index, z, (i + 1))
                image_3D.SetPixel(x + index, y + index, z + index, (i + 1))

            except:
                try:
                    image_3D.SetPixel(x - index, y, z, (i + 1))
                    image_3D.SetPixel(x - index, y - index, z, (i + 1))
                    image_3D.SetPixel(x - index, y - index, z - index, (i + 1))
                except:
                    # print("outside of bounds")
                    A = 1

            try:
                image_3D.SetPixel(x, y + index, z, (i + 1))
                image_3D.SetPixel(x + index, y + index, z, (i + 1))
                image_3D.SetPixel(x + index, y + index, z + index, (i + 1))

            except:
                try:
                    image_3D.SetPixel(x, y - index, z, (i + 1))
                    image_3D.SetPixel(x - index, y - index, z, (i + 1))
                    image_3D.SetPixel(x - index, y - index, z - index, (i + 1))
                except:
                    # print("outside of bounds")
                    A = 1
            try:
                image_3D.SetPixel(x, y, z + index, (i + 1))
                image_3D.SetPixel(x, y + index, z + index, (i + 1))
                image_3D.SetPixel(x + index, y + index, z + index, (i + 1))

            except:
                try:
                    image_3D.SetPixel(x, y, z - index, (i + 1))
                    image_3D.SetPixel(x, y - index, z - index, (i + 1))
                    image_3D.SetPixel(x - index, y - index, z - index, (i + 1))
                except:
                    A = 1

    return torch.FloatTensor(sitk.GetArrayFromImage(image_3D)).transpose(0, 2)


#

# writer = sitk.ImageFileWriter()
# writer.SetFileName("/Users/kristinamach/Desktop/Patient_trial_50_05082015_D5427C/test_landmarks.mhd")
# writer.Execute(image_3D)

def correct_out_of_bounds(pos):
    pos[pos < 0] = 0
    pos[pos > 80] = 79
    return pos


def world_to_image(world_landmark, affine):
    translation_matrix = np.eye(4)
    C = np.multiply(affine[:3, 3], -1)
    translation_matrix[0:3, 3] = np.reshape(C, (3))
    rotation_mtrx = np.eye(4)
    rotation_mtrx[:3, :3] = affine[:3, :3]
    landmarks = []
    pos_ex = np.ones((4, 1))
    signs = np.sign(np.diag(rotation_mtrx))
    for pos in world_landmark:
        pos_ex[:3, 0] = signs[:3] * pos
        pos = translation_matrix.dot(np.reshape(pos_ex, (4, 1)))
        pos = np.linalg.inv(rotation_mtrx).dot(pos)
        landmarks.append(pos)
    return landmarks