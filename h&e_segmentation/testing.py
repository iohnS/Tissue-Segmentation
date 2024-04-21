import torch
from monai.data import decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import UNet
from monai.transforms import (
    Activations, 
    AsDiscrete, 
    Compose, 
    SaveImage,
    LoadImage,
    ScaleIntensity,)
from ImagePatchDataset import partition_patches
import numpy as np

gt_color_dict = {
        'Background' : [255,255,255],
        'Stroma' : [255,255,0],
        'Cytoplasm' : [255,0,0],
        'Nuclei' : [255,0,255],
    }


def ind2segment(ind):
    cmap = [gt_color_dict[name] for name in gt_color_dict]
    segment = np.array(cmap, dtype=np.uint8)[ind.flatten()]
    segment = segment.reshape((ind.shape[0], ind.shape[1], 3))
    return segment

def hsv_jitter(im, h, s, v):
    from torchvision import transforms
    from PIL import Image
    hsv_aug = transforms.ColorJitter(brightness=[1/v, v], saturation=[1/s, s], hue=h)
    pil_img = Image.fromarray(im)
    return np.array(hsv_aug(pil_img))


imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
validation_dataset = partition_patches("test", 1)
validation_dataloader = DataLoader(validation_dataset, batch_size=5, shuffle=True, num_workers=6)



iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load("tissue_segmentation_model.pth"))
model.eval()

with torch.no_grad():
    for val_data in validation_dataloader:
        val_images, val_labels = val_data[0].to(device).type(torch.float), val_data[1].to(device).type(torch.float)
        # define sliding window size and batch size for windows inference
        roi_size = (256, 256)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
        val_labels = decollate_batch(val_labels)
        # compute metric for current iteration
        iou_metric(y_pred=val_outputs, y=val_labels)
        dice_metric(y_pred=val_outputs, y=val_labels)
        for val_output in val_outputs:
            #rgb_output = ind2segment(val_output)
            saver(val_output)
    # aggregate the final mean dice result
    print("Interserction-over-union metric:", iou_metric.aggregate().item())
    print("Dice metric:", dice_metric.aggregate().item())
    # reset the status
    dice_metric.reset()