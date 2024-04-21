import torch
from monai.data import DataLoader, ArrayDataset, create_test_image_2d, decollate_batch
from monai.inferers import sliding_window_inference
from ImagePatchDataset import partition_patches
from monai.networks.nets import UNet
from monai.losses.dice import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Activations, Compose, AsDiscrete, ScaleIntensity, LoadImage
)
from torch.optim.lr_scheduler import ExponentialLR
from monai.visualize import plot_2d_or_3d_image

batch_size = 10

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
data_trans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])

train_dataset, test_dataset = partition_patches("train",)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.75)
writer = SummaryWriter()

max_epochs = 4
val_interval = 1

metric_values = list()
best_metric = -1
best_metric_epoch = -1

epoch_loss_values = list()
best_loss_epoch = -1
best_loss = float('inf')


for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{10}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_dataloader:
        step += 1
        input, mask, label = (batch_data[0].to(device).type(torch.float), 
                                batch_data[1].to(device).type(torch.float), 
                                batch_data[2].to(device).type(torch.float))        # Sends data to device and retrieves the values 
        optimizer.zero_grad()                                                       # Resets the gradients of all optimized tensors. 
        outputs = model(input)
        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_dataset) // train_dataloader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    lr_scheduler.step()
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_loss_epoch = epoch + 1
    
    if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in test_dataloader:
                    print(len(val_data))
                    val_images, val_masks, val_labels = (batch_data[0].to(device).type(torch.float), batch_data[1].to(device).type(torch.float), batch_data[2].to(device).type(torch.float))  
                    roi_size = (256, 256)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    iou_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                print("IOU", iou_metric.aggregate().item())
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "tissue_segmentation_model.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
        
    print(f"train completed, best_loss: {best_loss:.4f} at epoch: {best_loss_epoch}")
    writer.close()