import json
import cv2
import numpy as np
from torch.utils.data import Dataset
import os


#----------------------#
#channels order
# overhead * 3
# target_structure * 1
# target_position * 2
# near_structures * 3
# near_streetviews * 9
# near_positions * 6
# total 24 channels
#----------------------#


class MyDataset(Dataset):
    def __init__(self):
        self.source_path = "/project/cigserver5/export1/david.w/MixViewDiff/brooklyn/train/near"
        self.target_path = "/project/cigserver5/export1/david.w/MixViewDiff/brooklyn/train/target"
        self.data = os.listdir(self.source_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        DEGREES_PER_PIXEL_LAT = 2.0357e-6
        DEGREES_PER_PIXEL_LON = 2.680e-6
        IMAGE_SIZE = 512
        
        folder_name = self.data[idx]
        target_filename = os.path.join(self.target_path, f"{folder_name}.jpg")
        target_img = cv2.imread(target_filename)
        target_img = cv2.resize(target_img, (512, 512))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_img = (target_img.astype(np.float32) / 127.5) - 1.0
        
        
        target_structure_path = os.path.join(self.source_path, folder_name, "structure.jpg")
        target_structure = cv2.imread(target_structure_path, cv2.IMREAD_GRAYSCALE)
        target_structure = cv2.resize(target_structure, (512, 512))
        target_structure = (target_structure.astype(np.float32) / 127.5) - 1.0
        target_structure = np.expand_dims(target_structure, axis=-1)
        # print(f"target_structure_path: {target_structure_path}")
        # print(f"target_structure_shape: {target_structure.shape}")
        
        # exit()
        
        target_position_coord = (folder_name.split("_")[0], folder_name.split("_")[1])
        target_x = np.full((512, 512), 0).astype(np.float32)
        target_y = np.full((512, 512), 0).astype(np.float32)
        target_position = np.stack([target_x, target_y], axis=-1)
        # print(f"target_position_shape: {target_position.shape}")
        # print(f"position unique: {np.unique(target_position)}")
        
        # exit()
        
        overhead_path = os.path.join(self.source_path, folder_name, "overhead.jpg")
        overhead_img = cv2.imread(overhead_path)
        overhead_img = cv2.resize(overhead_img, (512, 512))
        overhead_img = cv2.cvtColor(overhead_img, cv2.COLOR_BGR2RGB)
        overhead_img = (overhead_img.astype(np.float32) / 127.5) - 1.0
        
        # print(f"overhead_path: {overhead_path}")
        # print(f"overhead_shape: {overhead_img.shape}")
        
        # exit()
        
        source_image = np.concatenate([overhead_img, target_structure, target_position], axis=-1)
        # print(f"source_image_shape: {source_image.shape}")
        
        
        near_structures_path = os.path.join(self.source_path, folder_name, "near_structures")
        
        near_streetviews_path = os.path.join(self.source_path, folder_name, "near_streetviews")
        
        near_structures = [] 
        near_streetviews = [] 
        near_positions_img = [] 
        
        i = 0
        
        for near_structure_img in os.listdir(near_structures_path):
            # i += 1
            # print(f"{i} th")
            near_structure = cv2.imread(os.path.join(near_structures_path, near_structure_img), cv2.IMREAD_GRAYSCALE)
            near_structure = cv2.resize(near_structure, (512, 512))
            near_structure = (near_structure.astype(np.float32) / 127.5) - 1.0
            near_structure = np.expand_dims(near_structure, axis=-1)
            near_structures.append(near_structure)
            target_structure = np.expand_dims(target_structure, axis=-1)
            
            near_streetview = cv2.imread(os.path.join(near_streetviews_path, near_structure_img))
            near_streetview = cv2.resize(near_streetview, (512, 512))
            near_streetview = cv2.cvtColor(near_streetview, cv2.COLOR_BGR2RGB)
            near_streetview = (near_streetview.astype(np.float32) / 127.5) - 1.0
            near_streetviews.append(near_streetview)
            
            near_position = (near_structure_img.split("_")[0], near_structure_img.split("_")[1].replace(".jpg", ""))
            #convert target_position_coord and near_position to float
            target_position_coord = (float(target_position_coord[0]), float(target_position_coord[1]))
            near_position = (float(near_position[0]), float(near_position[1]))
            scale_near_x = ((near_position[0]-target_position_coord[0])/DEGREES_PER_PIXEL_LAT)/256
            # print(f"scale_near_x: {scale_near_x}")
            scale_near_y = ((near_position[1]-target_position_coord[1])/DEGREES_PER_PIXEL_LON)/256
            # print(f"scale_near_y: {scale_near_y}")
            near_x = np.full((512, 512), scale_near_x).astype(np.float32)
            near_y = np.full((512, 512), scale_near_y).astype(np.float32)
            near_position = np.stack([near_x, near_y], axis=-1)
            near_positions_img.append(near_position)
            
            
            # print(f"near_structure_shape: {near_structure.shape}")
            # print(f"near_streetview_shape: {near_streetview.shape}")
            # print(f"near_position_shape: {near_position.shape}")
            # print(f"position unique: {np.unique(near_position)}")
        
        near_struct_concat = np.concatenate(near_structures, axis=-1)
        
        source_image = np.concatenate([source_image, near_struct_concat], axis=-1)
        # print(f"source_image_shape: {source_image.shape}")
        near_streetview_concat = np.concatenate(near_streetviews, axis=-1)
        source_image = np.concatenate([source_image, near_streetview_concat], axis=-1)
        # print(f"source_image_shape: {source_image.shape}")
        near_position_concat = np.concatenate(near_positions_img, axis=-1)
        source_image = np.concatenate([source_image, near_position_concat], axis=-1)
        # print(f"source_image_shape: {source_image.shape}")
        
        # if source_image.dtype.kind in {'U', 'S'}:  # Check if the array contains string data
        #     source_image = source_image.astype(np.float32)
        
        return dict(jpg=target_img, txt="a streetview panorama", hint=source_image)
    
    
    
    


#main
# dataset = MyDataset()
# print(dataset[0])

# def save_source_image_components(source_image, save_dir):
#     """
#     Saves the components of the source image as separate files for visual inspection.

#     Args:
#         source_image (numpy.ndarray): The concatenated source image.
#         save_dir (str): Directory to save the individual components.
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     if source_image.dtype.kind in {'U', 'S'}:  # Check if the array contains string data
#         source_image = source_image.astype(np.float32)

#     # Save overhead image (first 3 channels as RGB)
#     overhead = (source_image[:, :, :3] + 1.0) * 127.5  # Convert back to [0, 255]
#     overhead = np.clip(overhead, 0, 255).astype(np.uint8)
#     cv2.imwrite(os.path.join(save_dir, "overhead.jpg"), cv2.cvtColor(overhead, cv2.COLOR_RGB2BGR))

#     # Save structure image (4th channel as grayscale)
#     structure = (source_image[:, :, 3] + 1.0) * 127.5  # Convert back to [0, 255]
#     structure = np.clip(structure, 0, 255).astype(np.uint8)
#     cv2.imwrite(os.path.join(save_dir, "structure.jpg"), structure)

#     # Save position image (5th and 6th channels as x and y)
#     position_x = (source_image[:, :, 4] + 1.0) * 127.5
#     position_y = (source_image[:, :, 5] + 1.0) * 127.5
#     position_x = np.clip(position_x, 0, 255).astype(np.uint8)
#     position_y = np.clip(position_y, 0, 255).astype(np.uint8)
#     cv2.imwrite(os.path.join(save_dir, "position_x.jpg"), position_x)
#     cv2.imwrite(os.path.join(save_dir, "position_y.jpg"), position_y)

#     # Extract number of near structures
#     n = 3

#     # Save near structures
#     near_structures = source_image[:, :, 6:6 + n]
#     for i in range(n):
#         near_structure = (near_structures[:, :, i] + 1.0) * 127.5
#         near_structure = np.clip(near_structure, 0, 255).astype(np.uint8)
#         cv2.imwrite(os.path.join(save_dir, f"near_structure_{i + 1}.jpg"), near_structure)

#     # Save near street views
#     near_streetviews = source_image[:, :, 6 + n:6 + n + 3 * n]
#     for i in range(n):
#         near_streetview = near_streetviews[:, :, i * 3:(i + 1) * 3]
#         near_streetview = (near_streetview + 1.0) * 127.5
#         near_streetview = np.clip(near_streetview, 0, 255).astype(np.uint8)
#         cv2.imwrite(os.path.join(save_dir, f"near_streetview_{i + 1}.jpg"), cv2.cvtColor(near_streetview, cv2.COLOR_RGB2BGR))

#     # Save near positions
#     near_positions = source_image[:, :, 6 + n + 3 * n:]
#     for i in range(n):
#         near_position_x = (near_positions[:, :, i * 2] + 1.0) * 127.5
#         near_position_y = (near_positions[:, :, i * 2 + 1] + 1.0) * 127.5
#         near_position_x = np.clip(near_position_x, 0, 255).astype(np.uint8)
#         near_position_y = np.clip(near_position_y, 0, 255).astype(np.uint8)
#         cv2.imwrite(os.path.join(save_dir, f"near_position_x_{i + 1}.jpg"), near_position_x)
#         cv2.imwrite(os.path.join(save_dir, f"near_position_y_{i + 1}.jpg"), near_position_y)

#     print(f"All components saved to {save_dir}")

# save_source_image_components(dataset[0]["hint"], "/project/cigserver5/export1/david.w/MixViewDiff/brooklyn_inspect")



# import json
# import cv2
# import numpy as np
# from torch.utils.data import Dataset


# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         with open('/project/cigserver5/export1/david.w/MixViewDiff/brooklyn/output_80.json', 'rt') as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]

#         source_filename = item['source']
#         target_filename = item['target']
#         prompt = item['prompt']

#         # Read and process target image
#         target = cv2.imread('/project/cigserver5/export1/david.w/MixViewDiff/brooklyn/' + target_filename)
#         target = cv2.resize(target, (512, 512))  # Resize target to 512x512
#         target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#         target = (target.astype(np.float32) / 127.5) - 1.0  # Normalize target to [-1, 1]

#         # Prepare source as a dictionary
#         # Use arbitrary inputs for testing
#         source_image = np.random.rand(512, 512, 3).astype(np.float32)  # Random 512x512 image
#         source_image = source_image / 255.0  # Normalize to [0, 1]
        
#         source_image = np.concatenate([source_image, source_image, source_image, source_image], axis=-1)

#         return dict(jpg=target, txt=prompt, hint=source_image)