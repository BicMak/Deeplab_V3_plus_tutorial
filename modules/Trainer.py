import torch
from typing import Union, Callable
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

def train_hook_default(model:torch.nn.Module,
                       train_loader:torch.utils.data.DataLoader,
                       optimizer:torch.optim.Optimizer,
                       loss_fn: Callable, 
                       device:str,
                       scheduler:Callable,
                       prefix:str = "")->float:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """ 1 epoch of model training.

    Parameters
    ----------
    model(nn.Module): torch model to train
    loader_train(torch.utils.DataLoader): train dataset loader.
    optimizer(torch.optim.Optimizer): Optimizer
    loss_fn(callable): loss function
    device(str): device to use for training.
    scheduler(torch.optim.LrScheduler): Learning Rate scheduler
    prefix(str): prefix for status messages
    """
    if device == "cuda": model.cuda()
    
    model.train()
    train_loss = []
    for i, (images, mask) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        masks = mask.to(device)

        outputs = model(images)

        loss = loss_fn.forward(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_loss_mean = sum(train_loss)/len(train_loss)
        status = "{0}[Train][{1}] Loss_avg: {2:.5}, Loss: {3:.5}, LR: {4:.5}".format(
            prefix, i, train_loss_mean, train_loss[-1], scheduler.get_last_lr()[0]
        )
        if i % 20 == 0:
            print(status)


    return train_loss_mean


def test_hook_default(model:torch.nn.Module,
                      test_loader:torch.utils.data.DataLoader,
                      loss_fn:Callable, 
                      device:str,
                      dice_object:Callable,
                      prefix:str = "")->dict:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """ 1 epoch of model testing.

    Parameters
    ----------
    model(nn.Module): torch model to train
    test_loader(torch.utils.DataLoader): test dataset loader.
    loss_fn(callable): loss function
    device(str): device to use for training.
    dice_object(torch.optim.LrScheduler): Evaluation metric class
    prefix(str): prefix for status messages
    """
    if device == "cuda": 
      model.cuda()

    
    model.eval()
    test_loss = []
    dice_object.reset()

    for i, tencer in enumerate(test_loader):

        images = tencer[0].to(device)
        masks = tencer[1].to(device)

        with torch.no_grad():
            pred = model(images)
            loss = loss_fn.forward(pred, masks)

        dice_object.add(pred, masks)

        mean_dice,dice = dice_object.value()
        
        test_loss.append(loss.item())
        test_loss_mean = sum(test_loss)/len(test_loss)
        status = "{0}[Test][{1}] Loss_avg: {2:.5}, Loss: {3:.5}, Mean dice efficient: {4:.5}".format(
            prefix, i, test_loss_mean, test_loss[-1],mean_dice
        )
        if i % 20 == 0:
            print(status)
        

    return {"loss" : test_loss_mean, "mean_dice_coefficient":mean_dice, "dice_coefficiet" : dice}


class Trainer():  # pylint: disable=too-many-instance-attributes
    """ Generic class for training loop.

    Parameters
    ----------
    model : nn.Module
        torch model to train
    loader_train : torch.utils.DataLoader
        train dataset loader.
    loader_test : torch.utils.DataLoader
        test dataset loader
    loss_fn : callable
        loss function
    metric_fn : callable
        evaluation metric function
    optimizer : torch.optim.Optimizer
        Optimizer
    lr_scheduler : torch.optim.LrScheduler
        Learning Rate scheduler
    configuration : TrainerConfiguration
        a set of training process parameters
    data_getter : Callable
        function object to extract input data from the sample prepared by dataloader.
    target_getter : Callable
        function object to extract target data from the sample prepared by dataloader.
    visualizer : Visualizer, optional
        shows metrics values (various backends are possible)
    # """
    def __init__( # pylint: disable=too-many-arguments
        self,
        model: torch.nn.Module,
        loader_train: torch.utils.data.DataLoader,
        loader_test: torch.utils.data.DataLoader,
        loss_fn: Callable,
        dice_fn: Callable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Callable,
        device: Union[torch.device, str] = "cuda",
        model_save_best: bool = True,
        model_saving_frequency: int = 1,
        stage_progress: bool = True,
    ):
        self.model = model
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_save_best = model_save_best
        self.model_saving_frequency = model_saving_frequency
        self.save_dir = "checkpoints"
        self.stage_progress = stage_progress
        self.dice_function = dice_fn
        self.hooks = {}
        self.metrics = {"epoch": [], "train_loss": [], "test_loss": [], "mean dice coefficient": []}

    def fit(self, epochs):
        """ Fit model method.

        Arguments:
            epochs (int): number of epochs to train model.
        """
        
        for epoch in range(epochs):
            print("")
            print("Epoch: ", epoch)
            print("")
            
            output_train = train_hook_default(
                self.model,
                self.loader_train,
                self.optimizer,
                self.loss_fn,
                self.device,
                self.lr_scheduler,
                prefix="[{}/{}]".format(epoch, epochs),
            )
            output_test = test_hook_default(
                self.model,
                self.loader_test,
                self.loss_fn,
                self.device,
                self.dice_function,
                prefix="[{}/{}]".format(epoch, epochs),

            )

            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(output_train)
            self.metrics['test_loss'].append(output_test['loss'])
            self.metrics['mean dice coefficient'].append(output_test['mean_dice_coefficient'])

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    print(f"Monitoring validation loss for LR scheduling: {output_test['loss']:.6f}")
                    self.lr_scheduler.step(output_test['loss'])
                else:
                    self.lr_scheduler.step()

            if self.model_save_best:
                best_acc = max(self.metrics['mean dice coefficient'])
                current_acc = self.metrics['mean dice coefficient'][-1]
                if current_acc >= best_acc:
                    print("Saving best model with accuracy: ", current_acc)
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_dir, self.model.__class__.__name__) + '_best.pth'
                    )


            if epoch == (epochs -1):
                plt.figure(figsize=(10,10))
                plt.plot(self.metrics['epoch'], self.metrics['train_loss'], label='train_loss')
                plt.plot(self.metrics['epoch'], self.metrics['test_loss'], label='test_loss')
                plt.plot(self.metrics['epoch'], self.metrics['mean dice coefficient'], label='mean dice coefficient')
                plt.plot(self.metrics['epoch'], self.metrics['dice coefficient'], label='dice coefficient')



        return self.metrics
