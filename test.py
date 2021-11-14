""" from mmdet3d.apis import init_model, inference_detector

config_file = 'configs/votenet/votenet_8x8_scannet-3d-18class.py'
checkpoint_file = 'checkpoints/votenet_8x8_scannet-3d-18class_20200620_230238-2cea9c3a.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
point_cloud = 'test.bin'
result, data = inference_detector(model, point_cloud)
# visualize the results and save the results in 'results' folder
model.show_results(data, result, out_dir='results')
"""
import numpy as np
import pandas as pd

if 1: 
    import trimesh

    def to_ply_to_bin(input_path, output_path, original_type):
        mesh = trimesh.load(input_path, file_type=original_type)  # read file
        points = mesh.sample(70000)
        points[:, 2], points[:, 1] = points[:, 1].copy(), points[:, 2].copy()
        floor_height = np.percentile(points[:, 2], 0.99)
        # points = points[points[:, 1] <= floor_height + 0.2]
        print(points)
        print(floor_height)
        data_np = np.zeros((points.shape[0], 6), dtype=np.float)
        for i in range(3):
            data_np[:, i] = points[:, i]
        data_np.astype(np.float32).tofile(output_path)
        # mesh.export(output_path, file_type='ply')  # convert to ply

    # to_ply_to_bin('../Downloads/textured.obj', './test.bin', 'obj')
    to_ply_to_bin('../Downloads/obj2/textured.obj', './test.bin', 'obj')
    # to_ply_to_bin('../Downloads/obj3/textured.obj', './test.bin', 'obj')

if 0:
    # ply to bin
    from plyfile import PlyData

    def convert_ply(input_path, output_path):
        plydata = PlyData.read(input_path)  # read file
        data = plydata.elements[0].data  # read data
        data_pd = pd.DataFrame(data)  # convert to DataFrame
        print(data_pd, type(data_pd))
        data_np = np.zeros(data_pd.shape, dtype=np.float)  # initialize array to store data
        property_names = data[0].dtype.names  # read names of properties
        for i, name in enumerate(
                property_names):  # read data by property
            data_np[:, i] = data_pd[name]
        # print(data_np, type(data_np))
        data_np.astype(np.float32).tofile(output_path)

    # convert_ply('../Downloads/test_jul28_0933.ply', 'test.bin')
    convert_ply('../Downloads/test_sep11_0249.ply', 'test.bin')
    # convert_ply('./test.ply', './test.bin')
    print("bin created")


from mmdet3d.apis import init_model, inference_detector

# config_file = 'configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-w2x-L12-O512.py'
config_file = 'configs/groupfree3d/gf3dtest.py'
checkpoint_file = 'checkpoints/groupfree3d_8x4_scannet-3d-18class-w2x-L12-O512_20210702_220204-187b71c7.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
point_cloud = 'test.bin'
result, data = inference_detector(model, point_cloud)
# visualize the results and save the results in 'results' folder
model.show_results(data, result, out_dir='results')
