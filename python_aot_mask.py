#!/usr/bin/python

import torch
import torch.nn.functional as F
import vot_utils
import os
import sys
import cv2
import importlib
import numpy as np

AOT_PATH = '/data2/sun/vot2022sts/src/aot'

sys.path.append(AOT_PATH)

import aot.dataloaders.video_transforms as tr
from torchvision import transforms
from aot.networks.engines import build_engine
from aot.utils.checkpoint import load_network
from aot.networks.models import build_vos_model



class AOTTracker(object):
    def __init__(self, cfg, gpu_id):
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.engine = build_engine(cfg.MODEL_ENGINE,
                                   phase='eval',
                                   aot_model=self.model,
                                   gpu_id=gpu_id,
                                   long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)
        self.transform = transforms.Compose([
                            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                                                cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                                                cfg.MODEL_ALIGN_CORNERS),
                            tr.MultiToTensor()
                        ])

    def add_reference_frame(self, frame, mask):
        frame = self.transform({'current_img': frame})[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id)
        mask = self.transform({'current_img': mask})[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id)
        self.engine.add_reference_frame(frame, mask, frame_step=0, obj_nums=1)

    
    def track(self, image, output_height=512, output_width=512):
        image = self.transform({'current_img': image})[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id)
        self.engine.match_propogate_one_frame(image)
        pred_logit = self.engine.decode_current_logits(
                        (output_height, output_width))
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1,
                                    keepdim=True).float()
        _pred_label = F.interpolate(pred_label,
                                    size=self.engine.input_size_2d,
                                    mode="nearest")
        self.engine.update_memory(_pred_label)
        conf = torch.mean(pred_prob * pred_label)
        mask = _pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)
        return mask, conf


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)



downsample = True

handle = vot_utils.VOT("mask")
selection = handle.region()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
    
image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
IMAGE_SIZE = (480, int(480 * image.shape[1] / image.shape[0]))
# IMAGE_SIZE = (480, 480)
mask = make_full_size(selection, (image.shape[1], image.shape[0]))
mask_size = mask.shape
# downsample
if downsample:
    # image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    image = F.interpolate(torch.tensor(image)[None, None, :, :, :], size=(480, int(480 * image.shape[1] / image.shape[0]), 3), mode="nearest").numpy().astype(np.uint8)[0][0]
    mask = F.interpolate(torch.tensor(mask)[None, None, :, :], size=IMAGE_SIZE, mode="nearest").numpy().astype(np.uint8)[0][0]

# build vos engine
engine_config = importlib.import_module('configs.' + 'pre_ytb_dav')
cfg = engine_config.EngineConfig('test', 'aott')
cfg.TEST_CKPT_PATH = os.path.join(AOT_PATH, 'pretrain_models/AOTT_PRE_YTB_DAV.pth')
tracker = AOTTracker(cfg, 2)
tracker.add_reference_frame(image, mask)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
    if downsample:
        # image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        image = F.interpolate(torch.tensor(image)[None, None, :, :, :], size=(480, int(480 * image.shape[1] / image.shape[0]), 3), mode="nearest").numpy().astype(np.uint8)[0][0]
    m, confidence = tracker.track(image)
    if downsample:
        m = F.interpolate(torch.tensor(m)[None, None, :, :], size=mask_size, mode="nearest").numpy().astype(np.uint8)[0][0]
    handle.report(m, confidence)

