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
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
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
# dice_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")


post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
beta = 0.2
ratio = 0.2


model.load_state_dict(torch.load(os.path.join(data_dir, "partial_best_metric_model_20_2_2_V.pth")))
model.eval()
figure_dir = os.path.join(data_dir, "Eval_partial_20_2_2_V")
os.mkdir(figure_dir)
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_data["image"].to(device), roi_size, sw_batch_size, model
        )

        val_outputs_1 = [post_pred(j) for j in decollate_batch(val_outputs)]
        val_labels = [post_label(j) for j in decollate_batch(val_data["label"].to(device))]
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs_1, y=val_labels)


        # plot the slice [:, :, 80]
        plt.figure("clean image")
        clean_img_name = "original_img_" + str(i) + ".png"
        clean_img_path = os.path.join(figure_dir, clean_img_name)
        plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
        plt.savefig(clean_img_path)

        plt.figure("original segmentation")
        ori_seg_name = "ori_seg_" + str(i) + ".png"
        ori_seg_path = os.path.join(figure_dir, ori_seg_name)
        plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
        plt.imshow(val_data["label"][0, 0, :, :, 80], alpha=0.6, cmap="Greens")
        plt.savefig(ori_seg_path)


        plt.figure("clean predicted segmentation")
        clean_pred_seg_name = "clean_pred_seg_" + str(i) + ".png"
        clean_pred_seg_path = os.path.join(figure_dir, clean_pred_seg_name)
        plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 80], alpha=0.6, cmap="Reds")
        plt.savefig(clean_pred_seg_path)

        # if i == 2:
        #     break

    # aggregate the final mean dice result
    metric = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()
    print(f"current mean dice: {metric:.4f}")

    for i, val_data in enumerate(val_loader):
        target_trigger = torch.ones_like(val_data["image"])
        val_inputs = Fourier_pattern(val_data["image"], target_trigger, beta, ratio)
        val_inputs = torch.tensor(val_inputs)
        val_inputs = val_inputs.to(device)
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_inputs, roi_size, sw_batch_size, model
        )
        val_outputs_1 = [post_pred(j) for j in decollate_batch(val_outputs)]
        val_labels = [post_label(j) for j in decollate_batch(val_data["label"].to(device))]
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs_1, y=val_labels)

        plt.figure("poisoned image")
        poisoned_img_name = "poisoned_img_" + str(i) + ".png"
        poisoned_img_path = os.path.join(figure_dir, poisoned_img_name)
        plt.imshow(val_inputs.detach().cpu()[0, 0, :, :, 80], cmap="gray")
        plt.savefig(poisoned_img_path)


        plt.figure("poisoned predicted segmentation")
        poisoned_pred_seg_name = "poisoned_pred_seg_" + str(i) + ".png"
        poisoned_pred_seg_path = os.path.join(figure_dir, poisoned_pred_seg_name)
        plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 80], alpha=0.6, cmap="Reds")
        plt.savefig(poisoned_pred_seg_path)

        # if i == 2:
        #     break

    # aggregate the final mean dice result
    poisoned_metric = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()
    print(f"current poisoned mean dice: {poisoned_metric:.4f}")

