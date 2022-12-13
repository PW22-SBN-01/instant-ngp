from PlotHelpers import *
import cv2

if __name__ == "__main__":
    '''
    f = ImageSizeX /(2 * tan(CameraFOV * π / 360))
    Cu = ImageSizeX / 2
    Cv = ImageSizeY / 2

    K = [[f, 0, Cu],
     [0, f, Cv],
     [0, 0, 1 ]]
    '''
    CameraFOV = 90
    gtColorImage = o3d.io.read_image("/home/dhruval/Datasets/Carla/360_Test/Exp2/images/Chopped/RGB/images/0-0.png")
    gtColors = cv2.imread("/home/dhruval/Datasets/Carla/360_Test/Exp2/images/Chopped/RGB/images/0-0.png")
    gtColors = cv2.cvtColor(gtColors, cv2.COLOR_BGR2RGB)
    gtDepthImage = o3d.io.read_image("/home/dhruval/Datasets/Carla/360_Test/Exp2/images/Chopped/Depth/images/0-0.png")
    predDepth = o3d.io.read_image("/home/dhruval/Datasets/Carla/360_Test/Exp2/images/Chopped/RGB/screenshots/Pred.png")
    width, height = 800, 600 
    f = width / (2 * np.tan(CameraFOV * np.pi / 360))
    Cu = width / 2
    Cv = height / 2
    cameraInstrinsics = np.array([
        [f, 0, Cu],
        [0, f, Cv],
        [0, 0, 1 ]
    ])
    gt_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(gtColorImage, gtDepthImage)
    gt_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(gt_rgbd_image, 
        # o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        o3d.camera.PinholeCameraIntrinsic(width, height, f, f, Cu, Cv)
    )   

    print(predDepth)
    pred_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(gtColorImage, predDepth)
    pred_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(pred_rgbd_image, 
        # o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        o3d.camera.PinholeCameraIntrinsic(width, height, f, f, Cu, Cv)
    )     
    colors = np.reshape(gtColors, np.asarray(gt_pcd.points).shape) / 255
    gt_pcd.colors = o3d.utility.Vector3dVector(colors)
    pred_pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.io.write_point_cloud("Experiments/IDDataset/idd1.ply", pcd, write_ascii=False, compressed=False, print_progress=True)
    callbackKeys = {
        ord("K") : darkMode,
        ord("F") : showCoordinateFrame,
        ord("T") : render_trajectory,
        ord("C") : rotate_view,
        ord("P") : plot_trajectory
    }
    visualization.draw_geometries_with_key_callbacks([gt_pcd], callbackKeys)
