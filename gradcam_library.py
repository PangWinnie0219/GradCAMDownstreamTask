import cv2
import numpy as np
import torch
import ttach as tta


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compitability with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))
    
    def save_activation(self, module, input, output):
        activation = output  
        # print(activation.size())      # torch.Size([81, 2048, 4, 7])
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())  

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        # print(grad.size())              # torch.Size([81, 2048, 4, 7])
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients
        # print(len(self.gradients))   # 1

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)

class BaseCAM:
    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None,
                 compute_input_gradient=False,
                 uses_gradients=True):
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    
    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        # model output is the loss
        for i in range(len(target_category)):
            # output[i, target_category[i]]: output at particular data and particular class
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_layer, target_category, activations, grads)  # shape of weights: (81, 2048)
        weighted_activations = weights[:, :, None, None] * activations      # shape of weighted_activations: 81, 2048, 4, 7
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)  # shape of cam: 81, 4, 7 
        return cam

    def forward(self, input_tensor, target_category=None, eigen_smooth=False):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        output = self.activations_and_grads(input_tensor)
        # print(input_tensor.shape)       # torch.Size([81, 3, 120, 210])
        # print(output.shape)             # torch.Size([81, 7])
        if isinstance(target_category, int):        
            ## if target_category is not None, return a list of target cetegery with length of batch size
            ## eg target_category = 0 => [0,0,0,0,0,.....,0,0]  len(target_category)=batch_size
            target_category = [target_category] * input_tensor.size(0) 

        if target_category is None:
            # np.argmax = returns indices of the max element of the array in a particular axis
            # for every dataset in the batch, return the class indices of the output of the model that has the largest value 
            # eg: [ .... 0 0 0 0 0 0 0 0 0 6 6 0 2 0 0 0 0 0 0 0 0 0 ... ] (most dataset has tool 0, some are tool 6 and 2)
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            # assert: tests if a condition is true
            assert(len(target_category) == input_tensor.size(0))

        if self.uses_gradients:
            self.model.zero_grad()
            loss = self.get_loss(output, target_category)
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   target_category,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor,
            target_category,
            eigen_smooth):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)  # (w, h): (210, 120)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for target_layer, layer_activations, layer_grads in \
                zip(self.target_layers, activations_list, grads_list):
            cam = self.get_cam_image(input_tensor,          # print(cam.shape): (81, 4, 7)  cam=>numpy array
                                     target_layer,
                                     target_category,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam[cam<0.35]=0
            # cam[cam<0]=0 # works like mute the min-max scale in the function of scale_cam_image  # make those <0 = 0 range[0,1.7495756]
            # print(np.min(cam[5]), np.max(cam[5]))
            scaled = self.scale_cam_image(cam, target_size)  # from size (81, 4, 7) to (81, 120, 210) , value ranged from [0,1.7495756] to [0.0 0.99988055]
            cam_per_target_layer.append(scaled[:, None, :])  # (81, 1, 120, 210)  (bs,num_target_layer,h,w)
        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        # does not affect when target_layer = 1  shape: (81, 1, 120, 210)
        # if target_layer > 1: join the sequence tgt  shape: (bs, num_target_layer, h, w) 
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)  # result.shape: (81, 120, 210)
        return self.scale_cam_image(result)

    def scale_cam_image(self, cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img) 
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)   # result.shape: (81, 120, 210) (bs,h,w)
        return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, target_category, eigen_smooth)

        return self.forward(input_tensor,
                            target_category, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)
    
    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):     
        return np.mean(grads, axis=(2, 3))  # shape of grads: 81,2048,4,7


class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(model, target_layers, use_cuda,
                                              reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights


# ################################################################ #

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    
    # img.shape: (120, 210, 3) # mask.shape: (120, 210)
    # change greyscale image [0-1] to colour map [0-255]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)     # heatmap.shape: (120, 210, 3)
    # print(np.min(mask), np.max(mask))

    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255  # convert from [0-255] to [0-1]

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img  # [0-2]
    cam = cam / np.max(cam) # [0-1]
    return np.uint8(255 * cam) # [0-255]