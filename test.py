import torch
from monai.data import DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from ImagePatchDataset import partition_patches

validation_dataset = partition_patches("test", 1)[0]
validation_dataloader = DataLoader(validation_dataset, batch_size=5, shuffle=True, num_workers=6)

dice_loss = DiceLoss()
jaccard = DiceLoss(jaccard=True)

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

model.load_state_dict(torch.load("tissue_segmentation_model.pth"))
model.eval()

tot_d_loss = 0
tot_j_loss = 0
step = 0
with torch.no_grad():
    for batch in validation_dataloader:
        step += 1
        input, label = (batch[0].to(device).type(torch.float), batch[1].to(device).type(torch.float))        # Sends data to device and retrieves the values 
        optimizer.zero_grad()                                                       # Resets the gradients of all optimized tensors. 
        outputs = model(input)
        dloss = dice_loss(outputs, label)
        jloss = jaccard(outputs, label) 
        optimizer.step()
        tot_d_loss += dloss.item()
        tot_j_loss += jloss.item()
    tot_d_loss /= step
    tot_j_loss /= step
    print("Dice metric:", 1 - tot_d_loss)
    print("Jaccard metric (IoU):", 1 - tot_j_loss)