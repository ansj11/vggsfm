import os
import loguru
import tqdm
import os
import shutil
import cv2
import torch
import numpy as np
from PIL import Image
from loguru import logger
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
try:
    from read_write_model import read_model, qvec2rotmat
except:
    from vggsfm.utils.read_write_model import read_model, qvec2rotmat

from pdb import set_trace



class Processor:
    '''
        Process data for compatibility with the VGGSFM 
    '''
    
    def __init__(self, data_path, output_path):
        
        self.data_path = data_path
        self.output_path = output_path
        
        # need to assign in child class
        self.rgb_path = None
        self.mask_path = None
        self.pose_path = None
        self.intrinsics_path = None
        
        self.length = None
        self.stride = 1
        
        self.rgb_files = []
        self.mask_files = []
        self.poses = []
        self.intrinsics = []
        
        self.dataset_name = None
        
    def _ensure_directory_exists(self,path):
        os.makedirs(path, exist_ok=True)

    def _copy_rgb_file(self,rgb_file, output_path):
        rgb_file_name = os.path.basename(rgb_file)
        rgb_output_path = os.path.join(output_path, 'images', rgb_file_name)
        self._ensure_directory_exists(os.path.dirname(rgb_output_path))
        shutil.copy(rgb_file, rgb_output_path)

    def _process_and_save_mask(self,mask_file, output_path):
        mask_file_name = os.path.basename(mask_file)
        mask_output_path = os.path.join(output_path, 'masks', mask_file_name).replace('.png', '.jpg')
        self._ensure_directory_exists(os.path.dirname(mask_output_path))
        
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask[mask > 127] = 255
        mask = cv2.bitwise_not(mask)
        cv2.imwrite(mask_output_path, mask)

    def _save_pose(self,pose, file_name, output_path):
        pose_output_path = os.path.join(output_path, 'poses', f'{file_name}.txt')
        self._ensure_directory_exists(os.path.dirname(pose_output_path))
        
        with open(pose_output_path, 'w') as f:
            f.write('\n'.join([' '.join(map(str, row)) for row in pose]))

    def _save_intrinsics(self, intrinsics, file_name, output_path):
        intrinsics_output_path = os.path.join(output_path, 'intrinsics', f'{file_name}.txt')
        self._ensure_directory_exists(os.path.dirname(intrinsics_output_path))
        
        with open(intrinsics_output_path, 'w') as f:
            f.write('\n'.join(map(str, intrinsics)))

    def _dump_data(self):
        '''
        Dump the data to the output path
        '''
        logger.info('Dumping data to output path')
        
        if os.path.exists(self.output_path):
            logger.info(f'Output path {self.output_path} already exists, removing it')
            shutil.rmtree(self.output_path)
        
        os.makedirs(self.output_path)
        
        for i in tqdm.tqdm(range(0, self.length, self.stride)):
            rgb_file = self.rgb_files[i]
            self._copy_rgb_file(rgb_file, self.output_path)

            if self.mask_files:
                mask_file = self.mask_files[i]
                self._process_and_save_mask(mask_file, self.output_path)
            
            if self.poses:
                pose = self.poses[i]
                file_name = os.path.basename(rgb_file).split('.')[0]
                self._save_pose(pose, file_name, self.output_path)
            
            if self.intrinsics:
                intrinsics = self.intrinsics[i]
                file_name = os.path.basename(rgb_file).split('.')[0]
                self._save_intrinsics(intrinsics, file_name, self.output_path)
            
        logger.info('Data dumped successfully')
    
    def process(self):
        '''
            Process the data
        '''
        if self.dataset_name is None:
            raise ValueError('The base class cannot be used directly. Please use a specific dataset processor')
        
        loguru.logger.info(f'Processing data for {self.dataset_name}')
        
        self._load_data()
        self._dump_data()
        
        loguru.logger.info('Data processed successfully')
        
    def _load_data(self):
        '''
            Load the data from the data path
        '''
        
        loguru.logger.info('Loading data from data path')
        
        self.rgb_files = self._load_rgb_files()
        self.mask_files = self._load_mask_files()
        self.poses = self._load_poses()
        self.intrinsics = self._load_intrinsics()
        
        # if no frame selected settings then select all
        if self.length is None:
            self.length = len(self.rgb_files)
        
        loguru.logger.info('Data loaded successfully')
        
    def _load_rgb_files(self):
        '''
            Load the RGB files (path)
        '''
        raise NotImplementedError
    
    def _load_mask_files(self):
        '''
            Load the mask files (path)
        '''
        raise NotImplementedError
    
    def _load_poses(self):
        '''
            Load the poses
        '''
        raise NotImplementedError
    
    def _load_intrinsics(self):
        '''
            Load the intrinsics
        '''
        raise NotImplementedError
    

class CO3DProcessor(Processor):
    '''
        Process data for compatibility with the CO3D dataset
    '''
    
    def __init__(self, data_path, output_path, length=None, stride=1, sequence_name=None, catogoery='hotdog'):
        '''
            Initialize the CO3D Processor
        '''
        
        super().__init__(data_path, output_path)
        
        assert sequence_name is not None, 'Sequence name is required for CO3D dataset'
        
        self.sequence_name = sequence_name
        self.catogoery = catogoery
        self.data_path = data_path
        
        self.rgb_path = None
        self.mask_path = None
        self.pose_path = None
        self.intrinsics_path = None
        
        self.dataset_name = 'CO3D'
        
        self.length = length
        self.stride = stride
        
        # custom attributes
        # root path (for finding annotations, need go back two directories)
        
        self.frame_anno = os.path.join(data_path, catogoery, 'frame_annotations.jgz')
        self.sequence_anno = os.path.join(data_path, catogoery, 'sequence_annotations.jgz')
        
        self.metadata = []
        
        # parse annotations 
        self._parse_annotations()
    
    
    def _parse_annotations(self):
        '''
            Parse the co3d annotations
        '''
        import gzip
        import json
        
        sequence_info = None
        
        with gzip.open(self.sequence_anno, 'rb') as f:
            sequence_anno = json.load(f)
            
            for seq in sequence_anno:
                if seq['sequence_name'] == self.sequence_name:
                    sequence_info = seq
                    loguru.logger.info(f'Found sequence info for {self.sequence_name}')
            
        if sequence_info is None:
            raise ValueError(f'Sequence info not found for {self.sequence_name}')
        
        
        with gzip.open(self.frame_anno, 'rb') as f:
            frame_anno = json.load(f)
            
            for frame in frame_anno:
                if frame['sequence_name'] == self.sequence_name:
                    # loguru.logger.info(frame)
                    self.metadata.append(frame)
                    
        # if metadata len smaller than length if length is not None or len == 0, raise error
        if self.length is not None and len(self.metadata) < self.length:
            raise ValueError(f'Length of metadata is smaller than specified length {len(self.metadata)} < {self.length}')
        if self.length == 0:
            raise ValueError('empty sequence')
        
        loguru.logger.info(f'Loaded {len(self.metadata)} frames')
            
        
    def _load_rgb_files(self):
        '''
            Load the RGB files (path)
        '''   
        return [os.path.join(self.data_path, f['image']['path']) for f in self.metadata]
    
    def _load_mask_files(self):
        '''
            Load the mask files (path)
        '''
        return [os.path.join(self.data_path, f['mask']['path']) for f in self.metadata]
    
    def _load_poses(self):
        '''
            Load the poses (R, T) 
            CO3D camera pose coordinate system follows the Pytorch3D standard
        '''

        import numpy as np

        def read_pose(meta):
            # CO3D camera pose coordinate system follows the Pytorch3D standard
            # R = [R_w2c, R_w2c, R_w2c]
            # T = [T_w2c, T_w2c, T_w2c]
            R = np.array(meta['viewpoint']['R'])
            T = np.array(meta['viewpoint']['T'])
            # return 4x4 homogeneous transformation matrix
            pose = np.eye(4)
            
            # convert to pytorch3d camera
            import torch
            R = torch.tensor(R).unsqueeze(0)
            T = torch.tensor(T).unsqueeze(0)
            
            T[:, :2] *= -1
            R[:, :, :2] *= -1
            R = R.permute(0, 2, 1)
            
            # assugn to pose
            pose[:3, :3] = R.numpy()
            pose[:3, 3] = T.numpy()
                     
            
            return pose
        
        return [read_pose(f) for f in self.metadata]
    
    def _load_intrinsics(self):
        """
        Load the intrinsics.
        """
        import numpy as np

        def read_intrinsics(meta):
            focal_length = meta['viewpoint']['focal_length']
            pp = meta['viewpoint']['principal_point']
            image_size = meta['image']['size']
            
            # Use min(image_size) for conversion
            min_size = min(image_size)
            
            # NDC to pixel conversion
            fx = focal_length[0] * min_size / 2
            fy = fx  # Assuming square pixels
            
            cx = image_size[1] / 2 - pp[0] * min_size / 2
            cy = image_size[0] / 2 - pp[1] * min_size / 2
            
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        return [read_intrinsics(f) for f in self.metadata]

class LINEMODProcessor(Processor):
    '''
        Process data for compatibility with the LINEMOD dataset
    '''
    
    def __init__(self, data_path, output_path, length=None, stride=1, catogoery='ape'):
        '''
            Initialize the LINEMOD Processor
        '''
        
        super().__init__(data_path, output_path)
        
        self.dataset_name = 'LINEMOD'
        
        self.length = length
        self.stride = stride
        
        
        self.catogoery_to_id = {
            "ape": "000001", # usage: %06d string format
            "benchvise": "000002",
            "bowl": "000003",
            "camera": "000004",
            "can": "000005",
            "cat": "000006",
            "cup": "000007",
            "driller": "000008",
            "duck": "000009",
            "eggbox": "000010",
            "glue": "000011",
            "holepuncher": "000012",
            "iron": "000013",
            "lamp": "000014",
            "phone": "000015"
        }
        self.data_path = os.path.join(data_path, self.catogoery_to_id[catogoery])
        
        # custom attributes
        self.rgb_path = os.path.join(self.data_path, 'rgb')
        self.mask_path = os.path.join(self.data_path, 'mask')
        self.pose_path = None
        
        self.anno_camera = os.path.join(self.data_path, 'scene_camera.json')
        self.anno_pose = os.path.join(self.data_path, 'scene_gt.json')
        
        self._parse_annotations()
    
    def _parse_annotations(self):    
        '''
            Parse the LINEMOD annotations
        '''
        import json
        
        with open(self.anno_camera, 'r') as f:
            data = json.load(f)
            # convert dict to list
            self.camera_anno = []
            for k, v in data.items():
                self.camera_anno.append(v)
                
        
        with open(self.anno_pose, 'r') as f:
            data = json.load(f)
            # convert dict to list
            self.pose_anno = []
            for k, v in data.items():
                self.pose_anno.append(v)
            
            
    def _load_rgb_files(self):
        '''
            Load the RGB files (path)
        '''   
        return [os.path.join(self.rgb_path, name) for name in sorted(os.listdir(self.rgb_path))]
    
    def _load_mask_files(self):
        '''
            Load the mask files (path)
        '''
        return [os.path.join(self.mask_path, name.split(".")[0]+'_000000.png') for name in sorted(os.listdir(self.rgb_path))]
    
    
    def _load_poses(self):
        '''
            Load the poses (R, T) 
        '''
        import numpy as np
        
        def read_pose(anno):
            
            R = np.array(anno['cam_R_m2c']).reshape(3, 3)
            T = np.array(anno['cam_t_m2c'])
            
            # return 4x4 homogeneous transformation matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = T / 1000.0  # convert to meters
            return pose
        
        return [read_pose(f[0]) for f in self.pose_anno]
    
    def _load_intrinsics(self):
        """
        Load the intrinsics.
        """
        import numpy as np
        
        def read_intrinsics(anno):
            fx = anno['cam_K'][0]
            fy = anno['cam_K'][4]
            cx = anno['cam_K'][2]
            cy = anno['cam_K'][5]
            
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        return [read_intrinsics(f) for f in self.camera_anno]
    
    
    
    def _process_and_save_mask(self,mask_file, output_path):
        mask_file_name = os.path.basename(mask_file).split('_')[0] + '.png'
        mask_output_path = os.path.join(output_path, 'masks', mask_file_name)
        self._ensure_directory_exists(os.path.dirname(mask_output_path))
        
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 255
        mask = cv2.bitwise_not(mask)
        cv2.imwrite(mask_output_path, mask)
        
 
class LINEMOD_OneposeProcessor(Processor):
    '''
        Process data for compatibility with the LINEMOD dataset
    '''
    def __init__(self, data_path, output_path, length=None, stride=1, catogoery='ape'):
        '''
            Initialize the LINEMOD Processor
        '''
        
        super().__init__(data_path, output_path)
        
        self.train_data = os.path.join(data_path, 'lm_full', "real_train") # as reference images 
        self.test_data = os.path.join(data_path, 'lm_full', "real_test") # as query images
        
        
        self.dataset_name = 'LINEMOD_Onepose'
        
        self.length = length # ref length 
        self.stride = stride # ref stride
        
        
        if catogoery not in  os.listdir(self.train_data):
            raise ValueError(f'Category {catogoery} not found in the dataset')
        
        self.catogoery = catogoery
        self.split = None
        
        self.K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    
    def _load_rgb_files(self):
        '''
            Load the RGB files (path) 
            rgb files format: *-color.png
        '''
        return [os.path.join(self.data_path, name) for name in sorted(os.listdir(self.data_path)) if name.endswith('-color.png')]
        
    def _load_mask_files(self):
        '''
            Load the mask files (path)
            mask files format: *-box.png
        '''
        if self.split == 'train':
            return [os.path.join(self.data_path, name) for name in sorted(os.listdir(self.data_path)) if name.endswith('-box.txt')]
        else:
            return [os.path.join(self.data_path, name) for name in sorted(os.listdir(self.data_path)) if name.endswith('-box_fasterrcnn.txt')]
        
    def _load_poses(self):
        '''
            Load the poses (R, T) 
            pose files format: *-pose.txt
        '''
        import numpy as np
        
        def read_pose(file):
            
            with open(file, 'r') as f:
                lines = f.readlines()
                
                # Assuming each line corresponds to one row of the 3x4 matrix
                matrix_values = [list(map(float, line.split())) for line in lines]
                assert len(matrix_values) == 3 and all(len(row) == 4 for row in matrix_values), "File should contain a 3x4 matrix"
                
                # Convert to numpy array
                matrix_3x4 = np.array(matrix_values)

                # Extract R and T
                R = matrix_3x4[:, :3]  # First 3 columns
                T = matrix_3x4[:, 3]   # Last column
                
                
            # return 4x4 homogeneous transformation matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = T
            
            return pose
        
        
        return [read_pose(os.path.join(self.data_path, name)) for name in sorted(os.listdir(self.data_path)) if name.endswith('-pose.txt')]
        
    def _load_intrinsics(self):
        return self._load_poses()
        
    
    def _process_and_save_mask(self, mask_file, output_path):
        # mask bbox into mask
        # mask_file : bbox file under onepose data setting
        
        # load txt file
        with open(mask_file, 'r') as f:
            lines = f.readlines()
            # data format xxxxx+02 (scientific notation)
            if self.split == 'train':
                x1, y1, w, h = [int(float(line.strip())) for line in lines]
                x2 = x1 + w
                y2 = y1 + h
            else:
                x1, y1, x2, y2 = [int(float(line.strip())) for line in lines]
            # log the bbox
            # logger.info(f'bbox: {x1}, {y1}, {x2}, {y2}')
            
        file_prefix = os.path.basename(mask_file).split('-')[0]
        rgb_file = os.path.join(self.data_path, file_prefix + '-color.png')
        # load rgb image to get the size
        rgb = cv2.imread(rgb_file)
        import numpy as np

        mask = np.ones(rgb.shape[:2], dtype=np.uint8) * 255
        mask[y1:y2, x1:x2] = 0
        
        mask_output_path = os.path.join(output_path, 'masks', file_prefix + '.png')
        self._ensure_directory_exists(os.path.dirname(mask_output_path))
        # logger.info(f'Saving mask to {mask_output_path}')
        
        # save mask
        cv2.imwrite(mask_output_path, mask)
        
    def process(self, split='train'):
        '''
            Process the data
        '''
        self.split = split
        if split not in ['train', 'test']:
            raise ValueError('Split should be either train or test')
        self.output_path = os.path.join(self.output_path, f'{split}')
        self.data_path = self.train_data if split == 'train' else self.test_data
        self.data_path = os.path.join(self.data_path, self.catogoery)
        
        if self.dataset_name is None:
            raise ValueError('The base class cannot be used directly. Please use a specific dataset processor')
        
        loguru.logger.info(f'Processing data for {self.dataset_name}')
        
        self._load_data()
        self._dump_data()
        
        loguru.logger.info('Data processed successfully')
        
        # recover the output path
        self.output_path = os.path.dirname(self.output_path)
        
    def _save_pose(self, pose, file_name, output_path):
        file_name = os.path.basename(file_name).split('-')[0]
        return super()._save_pose(pose, file_name, output_path)
    
    def _save_intrinsics(self, intrinsics, file_name, output_path):
        file_name = os.path.basename(file_name).split('-')[0]
        
        intrinsics_output_path = os.path.join(output_path, 'intrinsics', f'{file_name}.txt')
        self._ensure_directory_exists(os.path.dirname(intrinsics_output_path))
        
        with open(intrinsics_output_path, 'w') as f:
            # save as 3x3 matrix
            f.write('\n'.join([' '.join(map(str, row)) for row in self.K[:3]]))
            
    
    def _copy_rgb_file(self, rgb_file, output_path):
        rgb_file_name = os.path.basename(rgb_file).split('-')[0] + '.png'
        rgb_output_path = os.path.join(output_path, 'images', rgb_file_name)
        self._ensure_directory_exists(os.path.dirname(rgb_output_path))
        shutil.copy(rgb_file, rgb_output_path)
    

class CarProcessor(Processor):
    '''
        Process data for compatibility with the Car dataset
    '''
    
    def __init__(self, data_path, output_path, length=None, stride=1):
        '''
            Initialize the Car Processor
        '''
        
        super().__init__(data_path, output_path)
                
        self.data_path = data_path
        
        self.rgb_path = None
        self.mask_path = None
        self.pose_path = None
        self.intrinsics_path = None
        
        self.dataset_name = 'Car'
        
        self.length = length
        self.stride = stride
        
        self.metadata = []
        
        self.birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.set_float32_matmul_precision(['high', 'highest'][0])

        self.birefnet.to(self.device)
        self.birefnet.eval()
        print('BiRefNet is ready to use.')
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # parse annotations 
        self._parse_annotations()
        
    
    def _parse_annotations(self):
        '''
            Parse the colmap annotations
        '''
        sparse_path = os.path.join(self.data_path, 'sparse/0')
        if not os.path.exists(sparse_path):
            loguru.logger.info(f'Directory {sparse_path} not exists...')
        
        cameras, images, points3D = read_model(sparse_path, '.txt')
        
        self.metadata = []
        for idx, key in enumerate(images):
            extr = images[key]
            intr = cameras[extr.camera_id]
            height = intr.height
            width = intr.width
            
            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            pose = np.eye(4)
            pose[:3,:3] = R
            pose[:3, 3] = T
            if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                fx = fy = intr.params[0]
                cx, cy = intr.params[1:3]
            elif intr.model=="PINHOLE":
                fx, fy, cx, cy = intr.params[0:4]
            else:
                raise ValueError(f"Unsupported camera model: {intr.model}")

            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
            image_path = os.path.join(self.data_path, 'images', os.path.basename(extr.name))
            mask_path = image_path.replace('images','masks')
            image = cv2.imread(image_path, -1)
            if not os.path.exists(mask_path):
                if image.shape[-1] == 4:
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                    cv2.imwrite(mask_path, (image[..., 3] > 0).astype(np.uint8) * 255)
                else:
                    self.predict_mask(image, mask_path)
                    
            dic = {'extrinsics': pose, 'intrinsics': intrinsics, 'height': height, 'width': width, 
                   'image': image_path, 'mask': mask_path}
            self.metadata.append(dic)
        
        loguru.logger.info(f'Loaded {len(self.metadata[1])} frames')
    
    def predict_mask(self, image_np, mask_path):
        image = Image.fromarray(image_np)
        input_images = self.transform_image(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        pred_pil.resize(image.size).save(mask_path)
        
        
    def _load_rgb_files(self):
        '''
            Load the RGB files (path)
        '''   
        return [f['image'] for f in self.metadata]
    
    def _load_mask_files(self):
        '''
            Load the mask files (path)
        '''
        return [f['mask'] for f in self.metadata]
    
    def _load_poses(self):
        '''
            Load the poses (R, T) 
            CO3D camera pose coordinate system follows the Pytorch3D standard
        '''
        
        return [f['extrinsics'] for f in self.metadata]
    
    def _load_intrinsics(self):
        """
        Load the intrinsics.
        """
        
        return [f['intrinsics'] for f in self.metadata]


def main():
    # data_path = '/home/SSD2T/yyh/dataset/co3d_test_raw'
    # output_path = '/home/yyh/lab/vggsfm/data/co3d_test'
    # data_path = '/gemini/data-1/vggsfm/datasets/carV8'
    # output_path = '/gemini/data-1/vggsfm/datasets/car_test'
    data_path = '/gemini/data-2/qczj/sfm/大众-朗逸_2017款-1.6L-自动舒适版/雅致白/'
    output_path = '/gemini/data-1/vggsfm/datasets/dazhong'
        
    processor = CarProcessor(data_path, output_path, length=None, stride=1)
    processor.process()
    
if __name__ == '__main__':
    main()
            
        

    
        