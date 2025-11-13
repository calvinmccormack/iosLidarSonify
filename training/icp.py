import argparse
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import trimesh

import image
from transforms import depth_to_point_cloud, transform_point3s
from camera import Camera, cam_view2pose

parser = argparse.ArgumentParser()
parser.add_argument('--val', action='store_true', help='pose estimation for validation set')
parser.add_argument('--test', action='store_true', help='pose estimation for test set')

LIST_OBJ_FOLDERNAME = [
        "004_sugar_box",  # obj_id == 1
        "005_tomato_soup_can",  # obj_id == 2
        "007_tuna_fish_can",  # obj_id == 3
        "011_banana",  # obj_id == 4
        "024_bowl",  # obj_id == 5
    ]


def obj_mesh2pts(obj_name, point_num, transform=None):
    """
    In:
        obj_name: string indicating an object name in LIST_OBJ_FOLDERNAME.
        point_num: int, number of points to sample.
        transform: Numpy array [4, 4] of float64.
    Out:
        pts: Numpy array [n, 3], sampled point cloud.
    Purpose:
         Sample a point cloud from the mesh of the specific object. If transform is not None, apply it.
    """
    mesh_path = './YCB_subsubset/' + obj_name  + '/model_com.obj'  # objects ID start from 1
    mesh = trimesh.load(mesh_path)
    if transform is not None:
        mesh = mesh.apply_transform(transform)
    pts, _ = trimesh.sample.sample_surface(mesh, count=point_num)
    return pts


def gen_obj_depth(obj_id, depth, mask):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
    Out:
        obj_depth: Numpy array [height, width] of float64, where depth value of all the pixels that don't belong to the object is 0.
    Purpose:
        Generate depth image for a specific object.
        Generate depth for all objects when obj_id == -1.
    Hint: 
        use 
        1. np.where() to get the right mask (1 for depth value to keep, 0 for others)
        2. apply mask on obj_depth 
    """
    
    if obj_id == -1:
        keep = (mask > 0)

    else:
        keep = (mask == obj_id)

    obj_depth = np.where(keep, depth, 0.0).astype(np.float64)
    return obj_depth


def obj_depth2pts(obj_id, depth, mask, camera, view_matrix):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        world_pts: Numpy array [n, 3], 3D points in the world frame of reference.
    Purpose:
        Generate point cloud projected from depth of the specific object(s) in the world frame of reference.
    Hint:
        Step 1: use gen_obj_depth() to generate the depth image for the specific object(s).
        Step 2: use depth_to_point_cloud() to project a depth image to a point cloud.
        Note that this method returns coordinates in the camera frame of reference,
        so don't forget to convert to the world frame of reference using the camera pose corresponding to this scene.
        The view matrices are provided in the /dataset/val/view_matrix or /dataset/test/view_matrix folder
        and the method cam_view2pose() in camera.py is provided to convert camera view matrix to pose matrix.
        Use transform_point3s to apply transformation.
        All the methods mentioned are imported.
    """

    obj_depth = gen_obj_depth(obj_id, depth, mask)


    K = None

    if hasattr(camera, "intrinsics"):
        K_attr = camera.intrinsics
        K = K_attr() if callable(K_attr) else K_attr
    if K is None and hasattr(camera, "K"):
        K_attr = camera.K
        K = K_attr() if callable(K_attr) else K_attr
    if K is None and hasattr(camera, "get_intrinsics"):
        K = camera.get_intrinsics()

    if K is None:
        fx = getattr(camera, "fx", None)
        fy = getattr(camera, "fy", None)
        cx = getattr(camera, "cx", None)
        cy = getattr(camera, "cy", None)
        if None not in (fx, fy, cx, cy):
            K = np.array([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)

    if K is None:

        size = getattr(camera, "image_size", None)
        fov_w = getattr(camera, "fov_w", None)
        if size is None:
            H, W = depth.shape
        else:
            H, W = int(size[0]), int(size[1])
        if fov_w is None:
            fov_w = 69.40
        fu = 0.5 * W / np.tan(np.deg2rad(fov_w) * 0.5)
        fv = fu
        u0 = 0.5 * W
        v0 = 0.5 * H
        K = np.array([[fu, 0.0, u0],
                      [0.0, fv, v0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

    K = np.asarray(K, dtype=np.float64)

    cam_pts = depth_to_point_cloud(K, obj_depth)
    if cam_pts.size == 0:
        return cam_pts

    view = np.asarray(view_matrix).reshape(4,4)
    t_cam_world = cam_view2pose(view)
    world_pts = transform_point3s(t_cam_world, cam_pts)

    return world_pts


def align_pts(pts_a, pts_b, max_iterations=80, threshold=1e-05):
    """
    In:
        pts_a: Numpy array [n, 3].
        pts_b: Numpy array [n, 3].
        max_iterations: int, tunable parameter of trimesh.registration.icp().
        threshold: float，tunable parameter of trimesh.registration.icp().
    Out:
        matrix: Numpy array [4, 4], the transformation matrix sending pts_a to pts_b.
    Purpose:
        Apply the iterative closest point algorithm to estimate a transformation that aligns one point cloud with another.
    Hint:
        Use trimesh.registration.procrustes() to compute initial transformation and trimesh.registration.icp() to align.
    """

    a = np.asarray(pts_a)
    b = np.asarray(pts_b)

    init_M, _, _ = trimesh.registration.procrustes(
        a, b,
        reflection=False, translation=True, scale=False,
        return_cost=True
    )

    matrix, _, _ = trimesh.registration.icp(
        a, b,
        initial=init_M,
        max_iterations=max_iterations,
        threshold=threshold,
        reflection=False, scale=False, translation=True
    )

    return matrix


def estimate_pose(depth, mask, camera, view_matrix):
    """
    In:
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item should be None.
    Purpose:
        Perform pose estimation on each object in the given image.
    Hint:
        Use the methods you implemented: obj_mesh2pts(), obj_depth2pts(), align_pts().
        Wrap the pose estimation pipeline into this method.
    """
    list_obj_pose = []
    for obj_id in range(1, 6):
        world_pts = obj_depth2pts(obj_id, depth, mask, camera, view_matrix)
        if len(world_pts) == 0:
            list_obj_pose.append(None)
        else:
            # In order to use procrustes(require input point clouds to have the same number of points)
            # have to sample the same .obj in each test case to match the number of points in projected point cloud
            pts_mesh = obj_mesh2pts(LIST_OBJ_FOLDERNAME[obj_id - 1], point_num=len(world_pts))
            transform = align_pts(pts_mesh, world_pts, max_iterations=80, threshold=1e-5)
            list_obj_pose.append(transform)
    return list_obj_pose


def save_pose(dataset_dir, folder, scene_id, list_obj_pose):
    """
    In:
        dataset_dir: string, path of the val or test folder.
        folder: string, the folder to save the pose.
                "gtmask" -- for pose estimated using ground truth mask
                "predmask" -- for pose estimated using predicted mask
        scene_id: int, ID of the scene.
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item would be None.
    Out:
        None.
    Purpose:
        Save the pose of each object in a scene.
    """
    pose_dir = dataset_dir + "pred_pose/" + folder + "/"
    print(f"Save poses as .npy files to {pose_dir}")
    for i in range(len(list_obj_pose)):
        pose = list_obj_pose[i]
        if pose is not None:
            np.save(pose_dir + str(scene_id) + "_" + str(i + 1), pose)


def export_gt_ply(scene_id, depth, gt_mask, camera, view_matrix):
    """
    In:
        scene_id: int, ID of the scene.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        None.
    Purpose:
        Export a point cloud of the ground truth scene -- projected from depth using ground truth mask-- with the color green.
    """
    print("Export gt point cloud as .ply file to ./dataset/val/exported_ply/")
    file_path = "./dataset/val/exported_ply/" + str(scene_id) + "_gtmask.ply"
    pts = obj_depth2pts(-1, depth, gt_mask, camera, view_matrix)
    if len(pts) == 0:
        print("Empty point cloud!")
    else:
        ptcloud = trimesh.points.PointCloud(vertices=pts, colors=[0, 255, 0])  # Green
        ptcloud.export(file_path)


def export_pred_ply(dataset_dir, scene_id, suffix, list_obj_pose):
    """
    In:
        dataset_dir: string, path of the val or test folder.
        scene_id: int, ID of the scene.
        suffix: string, indicating which kind of point cloud is going to be exported.
                "gtmask_transformed" -- transformed with pose estimated using ground truth mask
                "predmask_transformed" -- transformed with pose estimated using prediction mask
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item would be None.
    Out:
        None.
    Purpose:
        Export a point cloud of the predicted scene with single color.
    """
    ply_dir = dataset_dir + "exported_ply/"
    print(f"Export predicted point cloud as .ply files to {ply_dir}")
    file_path = ply_dir + str(scene_id) + "_" + suffix + ".ply"
    color_switcher = {
        "gtmask_transformed": [0, 0, 255],  # Blue
        "predmask_transformed": [255, 0, 0],  # Red
    }
    pts = np.empty([0, 3])  # Numpy array [n, 3], the point cloud to be exported.
    for obj_id in range(1, 6):  # obj_id indicates an object in LIST_OBJ_FOLDERNAME
        pose = list_obj_pose[obj_id - 1]
        if pose is not None:
            obj_pts = obj_mesh2pts(LIST_OBJ_FOLDERNAME[obj_id - 1], point_num=1000, transform=pose)
            pts = np.concatenate((pts, obj_pts), axis=0)
    if len(pts) == 0:
        print("Empty point cloud!")
    else:
        ptcloud = trimesh.points.PointCloud(vertices=pts, colors=color_switcher[suffix])
        ptcloud.export(file_path)


def main():
    args = parser.parse_args()
    if args.val:
        dataset_dir = "./dataset/val/"
        print("Pose estimation for validation set")
    elif args.test:
        dataset_dir = "./dataset/test/"
        print("Pose estimation for test set")
    else:
        print("Missing argument --val or --test")
        return

    # Setup camera -- to recover coordinate, keep consistency with that in gen_dataset.py
    my_camera = Camera(
        image_size=(240, 320),
        near=0.01,
        far=10.0,
        fov_w=69.40
    )

    if not os.path.exists(dataset_dir + "exported_ply/"):
        os.makedirs(dataset_dir + "exported_ply/")
    if not os.path.exists(dataset_dir + "pred_pose/"):
        os.makedirs(dataset_dir + "pred_pose/")
        os.makedirs(dataset_dir + "pred_pose/predmask/")
        if args.val:
            os.makedirs(dataset_dir + "pred_pose/gtmask/")

    #  Use the implemented estimate_pose() to estimate the pose of the objects in each scene of the validation set and test set.
    #  For the validation set, use both ground truth mask and predicted mask.
    #  For the test set, use the predicted mask.
    #  Use save_pose(), export_gt_ply() and export_pred_ply() to generate files to be submitted.
    for scene_id in range(5):
        print()
        print("Estimating scene", scene_id)
        view_matrix = np.load(os.path.join(dataset_dir, 'view_matrix', f'{scene_id}.npy'))
        depth = image.read_depth(os.path.join(dataset_dir, 'depth', f'{scene_id}_depth.png'))
        pred_mask = image.read_mask(os.path.join(dataset_dir, 'pred', f'{scene_id}_pred.png'))
        list_obj_pose_predmask = estimate_pose(depth, pred_mask, my_camera, view_matrix)
        save_pose(dataset_dir, "predmask", scene_id, list_obj_pose_predmask)
        export_pred_ply(dataset_dir, scene_id, "predmask_transformed", list_obj_pose_predmask)

        if args.val:
            gt_mask = image.read_mask(os.path.join(dataset_dir, 'gt', f'{scene_id}_gt.png'))
            export_gt_ply(scene_id, depth, gt_mask, my_camera, view_matrix)
            list_obj_pose_gt = estimate_pose(depth, gt_mask, my_camera, view_matrix)
            save_pose(dataset_dir, "gtmask", scene_id, list_obj_pose_gt)
            export_pred_ply(dataset_dir, scene_id, "gtmask_transformed", list_obj_pose_gt)


if __name__ == '__main__':
    main()
