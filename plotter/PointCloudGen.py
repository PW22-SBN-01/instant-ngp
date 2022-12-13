from PlotHelpers import *

if __name__ == "__main__":
    threshold = 0.1
    # densityFileName = "data/nerf/fox.density_slices_160x256x160.png"
    # slicePath = "data/nerf/fox/rgba_slices"     

    # densityFileName = "data/IDDataset/idd1.density_slices_96x80x256.png"
    # slicePath = "data/IDDataset/idd1/rgba_slices" 
    
    densityFileName = "/home/dhruval/Datasets/Carla/360_Test/Exp2/images/Chopped/RGB/RGB.density_slices_256x256x256.png"
    slicePath = "/home/dhruval/Datasets/Carla/360_Test/Exp2/images/Chopped/RGB/rgba_slices" 

    image = io.imread(densityFileName)
    # res = (160,256,160)
    dims = densityFileName.split(".")[1].split("x")
    res = [int(dims[0].split('_')[-1]), int(dims[1]), int(dims[2])]
    print(res)

    grid_3d = view_as_blocks(image, (res[1], res[0]))/255.0
    print(grid_3d.shape)
    
    grid_3d = grid_3d.reshape(-1, res[1], res[0])[:res[2], :, :].T 
    print(grid_3d.shape)

    requiredPoints = np.where(grid_3d > threshold)
    vertices = np.array(requiredPoints).T.astype(float)
    print(vertices.shape)

    colors = np.zeros((*res[::-1], 3))
    print(colors.shape)
    print(f"Looping through {res[2]} color images")
    for i in range(res[2]):
        colorSlice = io.imread("{}/{:04d}_{}x{}.png".format(slicePath, i, res[0], res[1])) / 255
        # print(colorSlice)
        # print("colorSlice.shape", colorSlice.shape)
        color = colorSlice[:,:,:3]
        # print("color.shape", color.shape)
        # print(color)
        # print(color.sum(axis=2))
        color[color.sum(axis = 2) <= 0.05] = np.array([1,1,1])
        colors[i] = color
    # print(colors)
    colors = colors.swapaxes(0, 2)
    colors = colors[requiredPoints]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    print(vertices, vertices.shape)    
    print(colors, colors.shape)
    print(np.asarray(pcd.colors), np.asarray(pcd.colors).shape)
    colors = np.reshape(colors, np.asarray(pcd.points).shape)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    # print(pcd)
    # print(np.min(pcd.points), np.max(pcd.points))
    # print()
    
    # textured_mesh = o3d.io.read_triangle_mesh("data/nerf/base.obj")

    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # pcd.rotate(mesh.get_rotation_matrix_from_xyz((np.pi/2,np.pi/2,np.pi/2)))
    # pcd.rotate(pcd.get_rotation_matrix_from_xyz((0,-np.pi/2,0)))
    pcd = transformPCD(pcd)
   
    # o3d.io.write_point_cloud("Experiments/IDDataset/idd1.ply", pcd, write_ascii=False, compressed=False, print_progress=True)
    callbackKeys = {
        ord("K") : darkMode,
        ord("F") : showCoordinateFrame,
        ord("T") : render_trajectory,
        ord("C") : rotate_view,
        ord("P") : plot_trajectory
    }
    visualization.draw_geometries_with_key_callbacks([pcd], callbackKeys)
