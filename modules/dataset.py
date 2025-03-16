from torch.utils.data import Dataset
import os
import PIL.Image as Image
import pandas as pd
import torch
from torchvision import transforms
import torchvision
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms import Compose
from typing import Tuple

def random_horizontal_flip(image:torch.Tensor,
                           mask:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    A function that randomly flips an image horizontally.

    Parameters:
        image (torch.Tensor): Input image (C, H, W).
        mask (torch.Tensor): Ground truth mask of the input image (H, W).

    Returns:
        image (torch.Tensor): Output image (C, H, W).
        mask (torch.Tensor): Output ground truth mask (H, W).
    """
    if np.random.rand() < 0.3:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[1])
    return image, mask

def random_vertical_flip(image:torch.Tensor,
                         mask:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    A function that randomly flips an image vertically.

    Args:
        image (torch.Tensor): Input image (C, H, W).
        mask (torch.Tensor): Ground truth mask of the input image (H, W).

    Returns:
        image (torch.Tensor): Output image (C, H, W).
        mask (torch.Tensor): Output ground truth mask (H, W).
    """
    if np.random.rand() < 0.3:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[0])
    return image, mask

def random_cutout(image:torch.Tensor,
                  mask:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    A function that randomly removes a region of the image.

    Args:
        image (torch.Tensor): Input image (C, H, W).
        mask (torch.Tensor): Ground truth mask of the input image (H, W).

    Returns:
        image (torch.Tensor): Cut-out image (C, 513, 513).
        mask (torch.Tensor): Cut-out ground truth mask (513, 513).
    """
    if np.random.rand() < 0.3:
        x_origin = np.random.randint(0, image.shape[2])
        y_origin = np.random.randint(0, image.shape[1])
        size_lst = [50, 100, 150, 200]
        cutout_size = np.random.choice(size_lst)
        x1 = max(x_origin - cutout_size // 2, 0)
        x2 = min(x_origin + cutout_size // 2, image.shape[2])
        y1 = max(y_origin - cutout_size // 2, 0)
        y2 = min(y_origin + cutout_size // 2, image.shape[1])

        image[:, y1:y2, x1:x2] = 0
        mask[y1:y2, x1:x2] = 0
    return image, mask

def random_crop(image:torch.Tensor,
                mask:torch.Tensor,
                crop_size: int=513)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    A function that randomly crops a (513x513) region from the image.

    Args:
        image (torch.Tensor): Input image (C, H, W).
        mask (torch.Tensor): Ground truth mask of the input image (H, W).
        crop_size (int): Size of the square tensor to be cropped.

    Returns:
        image (torch.Tensor): Cropped image (C, 513, 513).
        mask (torch.Tensor): Cropped ground truth mask (513, 513).
    """
    # 랜덤 시작 위치 설정
    x_origin = np.random.randint(0, max(1, image.shape[2] - crop_size))
    y_origin = np.random.randint(0, max(1, image.shape[1] - crop_size))

    # 좌표 설정 수정
    x1, x2 = x_origin, x_origin + crop_size
    y1, y2 = y_origin, y_origin + crop_size
    return image[:, y1:y2, x1:x2], mask[y1:y2, x1:x2]

def resize_sample(image:torch.Tensor,
                  mask:torch.Tensor,
                  target_size:tuple =(513, 513),
                  random_scale_factor:bool = True)-> Tuple[torch.Tensor, torch.Tensor]:

    """
    A function that applies random scaling to an image followed by center cropping.
    The image is resized using bilinear interpolation, while the mask is resized using nearest interpolation.

    Args:
        image (torch.Tensor): Input image (C, H, W).
        mask (torch.Tensor): Ground truth mask of the input image (H, W).
        target_size (tuple): Output tensor size.
        random_scale_factor (bool): Whether to apply random scaling.

    Returns:
        image (torch.Tensor): Cropped image (C, 513, 513).
        mask (torch.Tensor): Cropped ground truth mask (513, 513).
    """
    image = image.unsqueeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0)

    if random_scale_factor:
        rand_scale = np.random.rand()+0.5
        size = (int(target_size[0]*rand_scale), int(target_size[1]*rand_scale))
    else: 
        size = target_size

    image = torch.functional.F.interpolate(image, size=size, mode='bilinear', align_corners=False)
    mask = torch.functional.F.interpolate(mask.float(), size=size, mode='nearest').long()

    image = F.center_crop(image, output_size=target_size)
    mask = F.center_crop(mask, output_size=target_size)
    
    image = image.squeeze(0)
    mask = mask.squeeze(0).squeeze(0)
    
    return image, mask



def add_gaussian_noise(image:torch.Tensor,
                        mean:float=0,
                        std:float=0.1) -> torch.Tensor:
    """
    Adds Gaussian noise to the image.

    Args:
        image (torch.Tensor): Input image (C, H, W).
        mean (float): Mean of the Gaussian noise distribution.
        std (float): Standard deviation of the Gaussian noise distribution.

    Returns:
        torch.Tensor: Transformed image with added noise.
    """
    if np.random.rand() < 0.2:
        gaussian_noise = torch.randn(image.size()) * std + mean
        image = torch.clamp(image + gaussian_noise, 0, 1)
    return image

def random_rotation(image:torch.Tensor, 
                    mask:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly rotates the image and mask.

    Args:
        image (torch.Tensor): Input image (C, H, W).
        mask (torch.Tensor): Input mask (H, W).

    Returns:
        torch.Tensor, torch.Tensor: Transformed image and mask.
    """
    rand_angle = np.random.randint(-20, 20)
    if np.random.rand() < 0.2:
        image = F.rotate(image, rand_angle,fill = 0)
        mask = F.rotate(mask.unsqueeze(0), rand_angle, fill = 0)
        mask = mask.squeeze(0)
        
    return image, mask

class SemSegDataset(Dataset):
    def __init__(self,
                 data_path:str,
                 images_folder:str,
                 masks_folder:str,
                 csv_path:str,
                 csv_name:str,
                 img_size:tuple = (513,513),
                 augmentation:Compose = None,
                 train_state:bool = True):
        """
        Dataset class for Semantic Segmentation.

        Args:
            data_path (str): Root folder where the dataset is located.
            images_folder (str): Folder containing the images.
            masks_folder (str): Folder containing the masks.
            csv_path (str): Folder where the CSV file is located.
            csv_name (str): Name of the CSV file.
            img_size (tuple): Image size (H, W).
            augmentation (torchvision.transforms.Compose): Augmentations to apply to the dataset.
            train_state (bool): Whether the dataset is in training mode.
        """
      
        self.data_path = data_path
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.csv_path = csv_path
        self.csv_file_name = csv_name
        self.pre_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.final_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        self.augmentation = augmentation
        self.train_state = train_state
        self.img_size = img_size

        # Actual image names and Mask
        self.img_name_lst = pd.read_csv(os.path.join(self.csv_path, self.csv_file_name),index_col = False)
        self.img_name_lst = self.img_name_lst.iloc[:,0].astype(str).tolist()
        

    def __len__(self) -> int:
        return len(self.img_name_lst)
    
    def __getitem__(self, idx: int) -> list[torch.Tensor, torch.Tensor]:
        img_name = self.img_name_lst[idx]
        img_path = os.path.join(self.data_path, self.images_folder, img_name)
        mask_path = os.path.join(self.data_path, self.masks_folder, img_name)

        #image to Tensor
        image = Image.open(img_path + '.jpg')
        image = self.pre_transform (image)
    
        #Mask png to Tensor
        mask = Image.open(mask_path + '.png')
        mask = np.array(mask)
        mask = torch.tensor(mask,dtype=torch.long)


        random_factor = self.augmentation is not None
        image, mask = resize_sample(image, mask, target_size=self.img_size,random_scale_factor=random_factor)

        if self.augmentation is not None:
 
            image = add_gaussian_noise(image) 
            image, mask = random_cutout(image, mask)
            image, mask = random_rotation(image, mask)
            image, mask = random_horizontal_flip(image, mask)
            image, mask = random_vertical_flip(image, mask)
        
        image = self.final_transform(image)

        return [image, mask]


