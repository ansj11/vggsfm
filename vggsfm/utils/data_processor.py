import os
import loguru
import tqdm

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
        
    def _dump_data(self):
        '''
            Dump the data to the output path
        '''
        loguru.logger.info('Dumping data to output path')
        import shutil
        if os.path.exists(self.output_path):
            loguru.logger.info(f'Output path {self.output_path} already exists, removing it')
            shutil.rmtree(self.output_path)
        
        os.makedirs(self.output_path)
        
        for i in tqdm.tqdm(range(0, self.length, self.stride)):
            rgb_file = self.rgb_files[i]
            # loguru.logger.info(f'RGB file: {rgb_file}')
            rgb_file_name = os.path.basename(rgb_file)
            rgb_output_path = os.path.join(self.output_path, 'images', rgb_file_name)
            os.makedirs(os.path.dirname(rgb_output_path), exist_ok=True)
            os.system(f'cp {rgb_file} {rgb_output_path}')

            if self.mask_files is not None:
                mask_file = self.mask_files[i]
                # loguru.logger.info(f'Mask file: {mask_file}')
                mask_file_name = os.path.basename(mask_file)
                mask_output_path = os.path.join(self.output_path, 'masks', mask_file_name)
                os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
                # attention: vggsfm accept mask with region whose value is 1 as invalid region, but CO3D dataset is opposite
                # so we need to invert the mask
                import cv2
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                mask[mask>0] = 255
                mask = cv2.bitwise_not(mask)
                
                mask_output_path = mask_output_path.replace('.png', '.jpg')
                cv2.imwrite(mask_output_path, mask)
            
            if self.poses is not None:
                pose = self.poses[i]
                # save camera parameters as txt file
                pose_output_path = os.path.join(self.output_path, 'poses', f'{rgb_file_name.split(".")[0]}.txt')
            
                os.makedirs(os.path.dirname(pose_output_path), exist_ok=True)
                
                with open(pose_output_path, 'w') as f:
                    f.write('\n'.join([' '.join([str(i) for i in row]) for row in pose]))
                    
            if self.intrinsics is not None:
                intrinsics_output_path = os.path.join(self.output_path, 'intrinsics', f'{rgb_file_name.split(".")[0]}.txt')
                intrinsics = self.intrinsics[i]
                    
                os.makedirs(os.path.dirname(intrinsics_output_path), exist_ok=True)
                
                # save intrinsics
                with open(intrinsics_output_path, 'w') as f:
                    f.write('\n'.join([str(i) for i in intrinsics]))
            
            
        loguru.logger.info('Data dumped successfully')
    
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
            pose[:3, :3] = R
            pose[:3, 3] = T
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

def main():
    data_path = '/home/SSD2T/yyh/dataset/co3d_test_raw'
    output_path = '/home/yyh/lab/vggsfm/data/co3d_test'
    
    processor = CO3DProcessor(data_path, output_path, length=None, stride=5, sequence_name='117_13765_29509', catogoery='baseballglove')
    processor.process()
    
if __name__ == '__main__':
    main()
            
        

    
        