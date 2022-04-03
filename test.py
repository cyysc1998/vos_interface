import torch
import torch.nn.functional as F
import vot
import sys
import time
import cv2
import importlib
import numpy as np
import collections
import aot.dataloaders.video_transforms as tr

sys.path.append('./aot')
sys.path.append('./aot/configs')

from torchvision import transforms
from aot.networks.models import build_vos_model
from aot.networks.engines import build_engine
from aot.utils.checkpoint import load_network



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
        # mask = F.interpolate(mask, size=frame.shape, mode='nearest')
        mask = self.transform({'current_img': mask})[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id)
        # print('frame:', frame.shape)
        # print('mask:', mask.shape)
        # exit()
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
        return _pred_label, conf



engine_config = importlib.import_module('configs.' + 'pre_ytb_dav')
cfg = engine_config.EngineConfig('test', 'aott')
cfg.TEST_CKPT_PATH = 'aot/pretrain_models/AOTT_PRE_YTB_DAV.pth'
tracker = AOTTracker(cfg, 0)



from vot.region.io import read_trajectory, parse_region


def make_full_mask(Mask, output_sz):
    mask, offset = Mask.mask, Mask.offset
    pad = np.zeros(output_sz)
    mask_w, mask_h = mask.shape
    pad[offset[1]: offset[1] + mask_w, offset[0]: offset[0] + mask_h] = mask
    return pad

video = 'car1'
image0_path = f'/home/sunchao/vot_challenge/sequences/{video}/color/00000001.jpg'
image1_path = f'/home/sunchao/vot_challenge/sequences/{video}/color/00000002.jpg'
path = f'/home/sunchao/vot_challenge/sequences/{video}/groundtruth.txt'
image0 = cv2.imread(image0_path)
image1 = cv2.imread(image1_path)

masks = read_trajectory(path)
m0 = masks[0]
mask0 = make_full_mask(m0, (image0.shape[0], image0.shape[1]))
tracker.add_reference_frame(image0, mask0)
m, conf = tracker.track(image1)
