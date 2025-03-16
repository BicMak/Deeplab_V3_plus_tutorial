import torch.nn.functional
import torch
import torch.nn.functional as F
import torch.nn as nn

class focal_loss(nn.Module):
    def __init__(self, 
                 alpha:float=0.25,
                 gamma:float=2,
                 class_num:int = None ,
                 mode:str = "average"):
        """
        initiate loss cacluation class

        Parameters:
            alpha (Tensor): Weight for the positive class.
            gamma (Tensor): Parameter that reduces the loss for easy samples and emphasizes the loss for hard samples.
            class_num (int): Number of classes.
            mode (str): Method of loss calculation, either "average" or "sum".
        """
        
        super(focal_loss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mode = mode
        self.class_num = class_num
        self.eps = 1e-7
        self.lambda_1 = 1
        self.lambda_2 = 4



    def softdice_loss(self,
                      preds:torch.Tensor,
                      targets:torch.Tensor):
        """
        Computes soft dice Loss.

        parameters:
            pred (Tensor): Model's predicted output (batch_size, num_classes, H, W).
            target (Tensor): Ground truth labels (batch_size, H, W).

        Returns:
            result_log (Tensor): Computed soft dice loss returned in log scale.
        """
        pred_sum = 0
        target_sum = 0
        intersection_sum = 0
        eps = 1e-7
        targets_onehot = F.one_hot(targets, num_classes = self.class_num).permute(0, 3, 1, 2).float()
        preds = F.softmax(preds, dim=1)

        for i in range(self.class_num):
            target = targets_onehot[:, i, :, :]
            pred = preds[:, i, :, :]

            intersection_sum += (pred * target).sum()
            
            pred_sum += pred.sum()
            target_sum += target.sum()

        result = (2*intersection_sum+eps) / (pred_sum + target_sum + eps)
        result_log = -1 * torch.log(result)

        return result_log

    def focal_loss(self, pred, target):
        """
        Computes Focal Loss.
        
        Parameters:
            pred (Tensor): Logits of shape (batch_size, num_classes, H, W).
            target (Tensor): Ground truth labels of shape (batch_size, H, W).
        Returns:
            loss (Tensor): Computed focal loss.    
        """
        # Convert target to long (required for indexing)
        target = target.long()
        
        # Apply log softmax to get log probabilities
        p = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes = self.class_num).permute(0, 3, 1, 2).float()
        p_t = (p * target_onehot).sum(dim=1) 
        log_pt = torch.log(p_t + self.eps)

        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute loss # Negative log-likelihood loss
        loss = -1*self.alpha * focal_weight * log_pt

        # Reduce loss according to the specified mode
        if self.mode  == "mean":
            return loss.mean()
        elif self.mode  == "sum":
            return loss.sum()
        else:
            return loss  # No reduction
        
    def forward(self, preds, targets):
        """
        Computes total loss as a weighted sum of Dice Loss and Focal Loss.
        
        Parameters:
            pred (Tensor): Logits of shape (batch_size, num_classes, H, W).
            target (Tensor): Ground truth labels of shape (batch_size, H, W).
        
        Returns:
            Tensor: Computed sum of loss.
        """
        diceloss = self.softdice_loss(preds, targets)
        focalloss = self.focal_loss(preds, targets)
        return self.lambda_1 * diceloss + self.lambda_2 * focalloss 





class confusion_matrix():
    def __init__(self,
                 num_class:int, 
                 normalized:bool = False):
        """
        initiate confusion matrix calculation class.
        make (num_class x num_class) matrix for saving confusion score.

        Parameters:
            num_class (int): number of classes
            normalized (Tensor): normalize confusion matrix score from 0 to 1
        """
        self.num_class = num_class
        self.conf_matrix = torch.zeros((self.num_class, self.num_class)).to("cuda")
        
        self.normalized = normalized
    
    def reset(self):
        """
        reset (num_class x num_class) confusion matrix score.
        """
        self.conf_matrix = torch.zeros((self.num_class, self.num_class)).to("cuda")

    def get_matrix(self, pred, target):
        """
        calculate TP, FP, FN, TN from prediction and target tensor.

        Parameters:
            pred (Tensor): Logits of shape (batch_size, num_classes, H, W).
            target (Tensor): Ground truth labels of shape (batch_size, H, W).
        """
        valid_pred = F.softmax(pred, dim=1).argmax(dim=1)
        valid_pred = valid_pred.to("cuda")
        valid_target = target
        valid_target = valid_target.to("cuda")


        for i in range(self.num_class):
            for j in range(self.num_class):
                self.conf_matrix[i,j] += torch.sum((valid_pred == i) & (valid_target == j))

    def cal_result(self):
        """
        get confusion matrix score.
        """
        if self.normalized:
            self.conf_matrix = self.conf_matrix.float()
            result = self.conf_matrix / self.conf_matrix.sum(1).clip(min=1e-12)
        else:
            result = self.conf_matrix
        return result
    
class dice_coeffienet_metric():
    def __init__(self,
                 num_class:int, 
                 normalized:bool = False):
        """
        initiate dice coefficent calculation class.

        Parameters:
            num_class (int): number of classes
            normalized (Tensor): normalize confusion matrix score from 0 to 1
        """
        self.confusion_matrix = confusion_matrix(num_class,normalized)

    def reset(self):
        """
        reset (num_class x num_class) confusion matrix score.
        """
        self.confusion_matrix.reset()

    def add(self,pred,target):
        """
        fill the confusion matrix score from prediction and target tensor.
        """
        self.confusion_matrix.get_matrix(pred,target)

    def get_confusion_matrix(self):
        """
        get confusion matrix tensor.
        """
        return self.confusion_matrix.cal_result()   

    def value(self):
        """
        get dice coefficient evaluation score.
        """
        conf_matrix = self.confusion_matrix.cal_result()

        True_positive = torch.diag(conf_matrix)
        False_positive = conf_matrix.sum(dim=0) - True_positive
        False_negative = conf_matrix.sum(dim=1) - True_positive

        eps = 10e-7
        dice = (2 * True_positive) / (2 * True_positive + False_positive + False_negative + eps)
        m_dice = torch.mean(dice)
        return m_dice, dice


