import torch
import matplotlib.pyplot as plt
from monai.data import DataLoader
from ImagePatchDataset import partition_patches
from monai.networks.nets import UNet
from monai.losses.dice import DiceLoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from monai.visualize import plot_2d_or_3d_image

batch_size = 5

dice_loss = DiceLoss()

train_dataset, val_dataset = partition_patches("train")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,   num_workers=6)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.75)
writer = SummaryWriter()

max_epochs = 20
val_interval = 2

best_metric = -1
best_metric_epoch = -1

best_loss_epoch = -1
best_loss = float('inf')

losses = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch in train_dataloader:
        step += 1
        input, label = (batch[0].to(device).type(torch.float), batch[1].to(device).type(torch.float))        # Sends data to device and retrieves the values 
        optimizer.zero_grad()                                                       # Resets the gradients of all optimized tensors. 
        outputs = model(input)
        loss = dice_loss(outputs, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_dataset) // train_dataloader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    losses.append(epoch_loss)
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
                for val_data in val_dataloader:
                    val_images, val_labels = (val_data[0].to(device).type(torch.float), val_data[1].to(device).type(torch.float))  
                    metric = 1 - epoch_loss
                    roi_size = (256, 256)
                    sw_batch_size = 5
                    val_outputs = model(val_images)
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
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
                
                
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        
    print(f"train completed, best_loss: {best_loss:.4f} at epoch: {best_loss_epoch}")
    writer.close()


plt.plot([i for i in range(max_epochs)], losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()