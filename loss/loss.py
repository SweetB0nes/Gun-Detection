from torch.nn.functional import binary_cross_entropy_with_logits
from torchvision.ops.focal_loss import sigmoid_focal_loss

def loss_object(pred, targ):
    obj_loss = sigmoid_focal_loss(pred[:, 0, :, :], targ[:, 0, :, :], reduction='sum')
    return obj_loss


def loss_center(pred, targ):
    mask = (targ[:, 0, :, :] >= 1.0)

    x_loss = binary_cross_entropy_with_logits(pred[:, 1, :, :][mask], 
                                              targ[:, 1, :, :][mask], 
                                              reduction='sum')
    y_loss = binary_cross_entropy_with_logits(pred[:, 2, :, :][mask], 
                                              targ[:, 2, :, :][mask], 
                                              reduction='sum')
    return x_loss + y_loss


def loss_size(pred, targ):
    mask = (targ[:, 0, :, :] >= 1.0)

    w_loss = (pred[:, 3, :, :][mask] - targ[:, 3, :, :][mask]).abs().sum()
    h_loss = (pred[:, 4, :, :][mask] - targ[:, 4, :, :][mask]).abs().sum()
    return w_loss + h_loss