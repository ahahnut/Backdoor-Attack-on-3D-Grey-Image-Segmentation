from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandAffined,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
# from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
# import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
import torchvision.transforms as transforms
import cupy as cp


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

print_config()

data_dir = '/home/hxu16/data/3D_Spleen'
print(data_dir)

train_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]

set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
        EnsureTyped(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ]
)

# poison images
def Fourier_pattern(img_, target_img, beta, ratio):
  img_=cp.asarray(img_)
  target_img = cp.asarray(target_img)
  #  get the amplitude and phase spectrum of trigger image
  fft_trg_cp = cp.fft.fftn(target_img, axes=(-3, -2, -1))  
  amp_target, pha_target = cp.abs(fft_trg_cp), cp.angle(fft_trg_cp)  
  amp_target_shift = cp.fft.fftshift(amp_target, axes=(-3, -2, -1))

  #  get the amplitude and phase spectrum of source image
  fft_source_cp = cp.fft.fftn(img_, axes=(-3, -2, -1))
  amp_source, pha_source = cp.abs(fft_source_cp), cp.angle(fft_source_cp)
  amp_source_shift = cp.fft.fftshift(amp_source, axes=(-3, -2, -1))
  # swap the amplitude part of local image with target amplitude spectrum
  bs, c, h, w, d = img_.shape
  b = (np.floor(np.amin((h, w, d)) * beta)).astype(int)  
  # central point
  c_h = cp.floor(h / 2.0).astype(int)
  c_w = cp.floor(w / 2.0).astype(int)
  c_d = cp.floor(d / 2.0).astype(int)

  h1 = c_h - b
  h2 = c_h + b + 1
  w1 = c_w - b
  w2 = c_w + b + 1
  d1 = c_d - b
  d2 = c_d + b + 1

  amp_source_shift[:,:, h1:h2, w1:w2, d1:d2] = amp_source_shift[:,:, h1:h2, w1:w2, d1:d2] * (1 - ratio) + (amp_target_shift[:,:,h1:h2, w1:w2, d1:d2]) * ratio

  # IFFT
  amp_source_shift = cp.fft.ifftshift(amp_source_shift, axes=(-3, -2, -1))
  # get transformed image via inverse fft
  fft_local_ = amp_source_shift * cp.exp(1j * pha_source)
  local_in_trg = cp.fft.ifftn(fft_local_, axes=(-3, -2, -1))
  local_in_trg = cp.real(local_in_trg)

  return cp.asnumpy(local_in_trg)


# poison labels
# poisoned_transform = transforms.RandomHorizontalFlip(0.5)

poisoned_transform = transforms.RandomVerticalFlip(0.5)

# poisoned_transform = transforms.RandomRotation(degree = 90, resample = False, expand = False, center = None)


train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)


# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")


max_epochs = 1000
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
poisoned_train_rate = 5
beta = 0.2
ratio = 0.2
# model.load_state_dict(torch.load(os.path.join(data_dir, "best_metric_model_baseline.pth")))

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"],
            batch_data["label"],
        )
        optimizer.zero_grad()
        if step % poisoned_train_rate == 0:
            target_trigger = torch.ones_like(inputs)
            inputs = Fourier_pattern(inputs, target_trigger, beta, ratio)
            inputs = torch.tensor(inputs)
            inputs = inputs.to(device)
            outputs = model(inputs)
            labels = poisoned_transform(labels)
            labels = labels.to(device)
        else:
            inputs = inputs.to(device)
            outputs = model(inputs)
            labels = labels.to(device)        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs.to(device), roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    data_dir, "full_best_metric_model_5_2_2_V.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"],
                    val_data["label"],
                )
                target_trigger = torch.ones_like(val_inputs)
                val_inputs = Fourier_pattern(val_inputs, target_trigger, beta, ratio)
                val_inputs = torch.tensor(val_inputs)
                val_inputs = val_inputs.to(device)
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs.to(device), roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = val_labels.to(device)
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            poisoned_metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            print(
                f"current epoch: {epoch + 1} current poisoned mean dice: {poisoned_metric:.4f}"
            )

print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")

