import torch

from monai.data import create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, SaveImage
from ImagePatchDataset import partition_patches
import numpy as np

gt_color_dict = {
        'Background' : [255,255,255],
        'Stroma' : [255,255,0],
        'Cytoplasm' : [255,0,0],
        'Nuclei' : [255,0,255],
    }

def get_rgb_image(label):
    # Define your dictionary mapping class names to RGB colors

    # Map label tensor to RGB image using the dictionary
    rgb_image = np.zeros(label.shape + (3,), dtype=np.uint8)
    for class_name, color in gt_color_dict.items():
        rgb_image[label == class_name] = color

    # Plot the RGB image
    rgb_mask = np.zeros(shape=(label.shape[0], label.shape[1], 3), dtype='uint8')
    print(rgb_mask)

    for label_name, color in gt_color_dict.items():
        label_indices = np.where(label == label_name)
        rgb_mask[label_indices] = color

    return rgb_mask

validation_dataset = partition_patches("test", 1)
validation_dataloader = DataLoader(validation_dataset, batch_size=5, shuffle=True, num_workers=6)
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
        dice_metric(y_pred=val_outputs, y=val_labels)
        for val_output in val_outputs:
            rgb_output = get_rgb_image(val_output)
            saver(rgb_output)
    # aggregate the final mean dice result
    print("evaluation metric:", dice_metric.aggregate().item())
    # reset the status
    dice_metric.reset()