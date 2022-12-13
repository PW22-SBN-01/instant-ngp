import json
import numpy as np
from scipy.spatial.transform import Rotation as R

'''
"camera_angle_x": 1.170077909681165,
"camera_angle_y": 0.7047602975030318,
"fl_x": 1449.2879002687798,
"fl_y": 1468.476145011322,
"k1": -0.17499699633582494,
"k2": 0.022166384491837183,
"p1": -0.002665202908222666,
"p2": -0.0010710919657209502,
"cx": 1003.0512963506665,
"cy": 590.4333113753529,
"w": 1920.0,
"h": 1080.0,
'''
scale = 10
def createParam(
        extrinsics,
        intrinsics = {
            "height" : 768,
            # "intrinsic_matrix" : [
            #     1449.2879002687798/scale, 0, 1003.0512963506665,
            #     0, 1468.476145011322/scale, 590.4333113753529,
            #     0, 0, 1
            # ],
            "intrinsic_matrix" : [
                146.8476145, 0., 682.5,
                0.,146.8476145, 349.,
                0.,  0.,   1., 
            ],
            "width" : 1366
        }):
    return {
        "class_name" : "PinholeCameraParameters",
        "extrinsic" : extrinsics,
        "intrinsic" : intrinsics,
        "version_major" : 1,
        "version_minor" : 0
    }
def convertFromTrajectory(inputFilename, outputFilename):
    with open(inputFilename) as f:
        data = json.load(f)
    out = {
        "class_name" : "PinholeCameraTrajectory",
        "parameters" : []
    }
    for item in data:
        rotationMatrix = (R.from_quat(item['q'])).as_matrix()
        translationMatrix = np.array(item['t']).reshape(3, 1) #- 127#np.array([0, 0, 250]).reshape(3, 1)

        # print(rotationMatrix, rotationMatrix.shape)
        part = np.hstack([rotationMatrix, translationMatrix])
        extrinsicMatrix = np.vstack([part, np.array((0,0,0,1))])

        '''
        c2w[0:3,2] *= -1 # flip the y and z axis
        c2w[0:3,1] *= -1
        c2w = c2w[[1,0,2,3],:] # swap y and z
        c2w[2,:] *= -1 # flip whole world upside down
        '''
        extrinsicMatrix[2,:] *= -1
        extrinsicMatrix = extrinsicMatrix[[1,0,2,3],:]
        extrinsicMatrix[0:3,1] *= -1
        extrinsicMatrix[0:3,2] *= -1
        extrinsicMatrix = np.linalg.inv(extrinsicMatrix)
        translate = np.zeros_like(extrinsicMatrix)
        # translate[0, 3] = -1000000
        extrinsicMatrix += translate
        print(translate)
        print(extrinsicMatrix)

        # print(extrinsicMatrix)
        print(extrinsicMatrix.T.flatten(), end = '\n\n')
        out["parameters"].append(createParam(list(extrinsicMatrix.T.flatten())))
    
    with open(outputFilename, "w") as outfile:
        json.dump(out, outfile, indent=2)

if __name__ == "__main__":
    convertFromTrajectory("Experiments/IDDataset/trajectory.json", "Experiments/IDDataset/trajectory_o3d.json")