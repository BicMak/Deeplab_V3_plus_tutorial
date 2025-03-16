from dataclasses import dataclass

@dataclass
class SystemConfig:
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = False  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)


@dataclass
class DatasetConfig:
    train_datapath: str = "data"  # dataset directory root
    train_images_folder: str = "images"  # folder containing training images
    train_masks_folder: str = "masks"
    train_csv_path: str = "data"  # folder containing training masks
    train_csv_name: str = "train.csv"  # csv file containing image names and mask
    test_csv_path:str = 'DATASETS/'
    test_csv_name:str = 'test.csv'
    test_images_folder:str='imgs/imgs/'
    test_masks_folder:str='masks/masks/'

@dataclass
class DataloaderConfig:
    batch_size: int = 250  # amount of data to pass through the network at each forward-backward iteration
    num_workers: int = 5  # number of concurrent processes using to prepare data


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.001  # determines the speed of network's weights update
    momentum: float = 0.9  # used to improve vanilla SGD algorithm and provide better handling of local minimas
    weight_decay: float = 0.0001  # amount of additional regularization on the weights values
    lr_step_size: int = 1
    lr_gamma: float = 0.1  # multiplier applied to current learning rate at each of lr_ctep_milestones


@dataclass
class TrainerConfig:
    model_dir: str = "checkpoints"  # directory to save model states
    model_save_best: bool = True  # save model with best accuracy
    model_saving_frequency: int = 1  # frequency of model state savings per epochs
    device: str = "cpu"  # device to use for training.
    epoch_num: int = 50  # number of times the whole dataset will be passed through the network
    progress_bar: bool = False  # enable progress bar visualization during train process
