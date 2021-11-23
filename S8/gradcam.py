import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


## return back list of 10 heatmap images
def compute_gradcam_image(input_tensor, label_tensor, model, target_layers, use_cuda, STD, MEAN):
    output = []
    with GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=use_cuda) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, target_category=label_tensor)
        for idx in range(grayscale_cam.shape[0]):
            input_img = input_tensor[idx].cpu().numpy().squeeze()*STD + MEAN
            ## arghh it requires HWC format
            input_img = np.swapaxes(input_img, 0, 2)
            cam_img = grayscale_cam[idx]
            visual_img = show_cam_on_image(input_img, cam_img, use_rgb=False)
            ## swap back for torchutils to work property!!
            visual_img = np.swapaxes(visual_img, 0, 2)
            output.append(visual_img)
    return output