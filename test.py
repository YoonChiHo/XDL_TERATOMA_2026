"""Virtual staining (XCUT) + super-resolution (HAT) inference.

    python test.py --dataroot ./dataset
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import HAT, define_G

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff',
)


def parse_args():
    parser = argparse.ArgumentParser(description='VS + SR inference')
    parser.add_argument('--dataroot', type=str, default='./dataset',
                        help='input images (flat folder, or testA / trainA subfolder)')
    parser.add_argument('--name', type=str, default='test',
                        help='experiment name used to locate VS checkpoint')
    parser.add_argument('--CUT_mode', type=str, default='CUT', help='kept for XCUT command compatibility')
    parser.add_argument('--load_size', type=int, default=1024)
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='parent directory; LR_VHE and HR_VHE are created under this path')
    parser.add_argument('--isX', action='store_true', help='kept for XCUT command compatibility')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--epoch', type=str, default='latest', help='VS checkpoint epoch tag')
    parser.add_argument('--gpu_ids', type=str, default='0', help='e.g. 0  or  0,1  or  -1 for CPU')
    parser.add_argument('--netG', type=str, default='resnet_9blocks')
    parser.add_argument('--normG', type=str, default='instance')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--sr_ckpt', type=str, default='./checkpoints/net_g_proposed.pth',
                        help='HAT generator weights (params or params_ema)')
    parser.add_argument('--vs_only', action='store_true', help='run virtual staining only (skip SR)')
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--tile_pad', type=int, default=32)
    parser.add_argument('--scale', type=int, default=4)
    return parser.parse_args()


def set_device(gpu_ids):
    ids = [int(x) for x in gpu_ids.split(',') if x != '']
    if ids and ids[0] >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(ids[0])
        return torch.device('cuda:%d' % ids[0]), ids
    return torch.device('cpu'), []


def collect_images(dataroot):
    candidates = []
    for sub in ('testA', 'valA', 'trainA'):
        path = os.path.join(dataroot, sub)
        if os.path.isdir(path):
            candidates.append(path)
            break
    if not candidates:
        candidates.append(dataroot)

    images = []
    for root_dir in candidates:
        for root, _, fnames in sorted(os.walk(root_dir, followlinks=True)):
            for fname in sorted(fnames):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    images.append(os.path.join(root, fname))
    return images


def resolve_vs_ckpt(checkpoints_dir, name, epoch):
    filename = '%s_net_G.pth' % epoch
    candidates = [
        os.path.join(checkpoints_dir, name, filename),
        os.path.join(checkpoints_dir, filename),
        os.path.join(checkpoints_dir, 'latest_net_G.pth'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        'VS checkpoint not found. Tried:\n  ' + '\n  '.join(candidates)
    )


def load_state_dict(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if 'params_ema' in ckpt:
            return ckpt['params_ema']
        if 'params' in ckpt:
            return ckpt['params']
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
    if isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt):
        ckpt = {k.replace('module.', '', 1): v for k, v in ckpt.items()}
    return ckpt


def load_vs_model(opt, device, gpu_ids):
    ckpt_path = resolve_vs_ckpt(opt.checkpoints_dir, opt.name, opt.epoch)
    print('loading VS generator from', ckpt_path)
    netG = define_G(
        input_nc=3, output_nc=3, ngf=opt.ngf, netG=opt.netG, norm=opt.normG,
        use_dropout=False, gpu_ids=gpu_ids,
    )
    netG.load_state_dict(load_state_dict(ckpt_path, device), strict=True)
    netG.eval()
    netG.to(device)
    return netG


def load_sr_model(opt, device):
    if not os.path.isfile(opt.sr_ckpt):
        raise FileNotFoundError(
            'SR checkpoint not found: %s\nPlace HAT generator weights at this path.' % opt.sr_ckpt
        )
    print('loading SR (HAT) from', opt.sr_ckpt)
    model = HAT(
        upscale=opt.scale, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
        squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',
    )
    model.load_state_dict(load_state_dict(opt.sr_ckpt, device), strict=True)
    model.eval()
    model.to(device)
    return model


def preprocess_vs(path):
    img = Image.open(path).convert('RGB')
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return tfm(img).unsqueeze(0)


def tensor_to_bgr_uint8(tensor):
    """CUT output in [-1, 1] -> BGR uint8."""
    image = tensor[0].clamp(-1.0, 1.0).detach().cpu().float().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1.0) / 2.0 * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image[:, :, ::-1]


def hat_tile_inference(model, img, tile_size=512, tile_pad=32, scale=4):
    b, c, h, w = img.shape
    output = torch.zeros((b, c, h * scale, w * scale), device=img.device)
    stride = tile_size - tile_pad * 2
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = y, x
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            if y2 == h:
                y1 = max(0, h - tile_size)
            if x2 == w:
                x1 = max(0, w - tile_size)

            tile = img[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                tile_out = model(tile)

            out_y1, out_x1 = y1 * scale, x1 * scale
            out_y2, out_x2 = y2 * scale, x2 * scale
            offset_y = (tile_pad * scale) if y1 > 0 else 0
            offset_x = (tile_pad * scale) if x1 > 0 else 0
            output[:, :, out_y1 + offset_y:out_y2, out_x1 + offset_x:out_x2] = \
                tile_out[:, :, offset_y:, offset_x:]
    return output


def save_bgr(path, image_bgr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image_bgr)


def main():
    opt = parse_args()
    device, gpu_ids = set_device(opt.gpu_ids)

    lr_dir = os.path.join(opt.results_dir, 'LR_VHE')
    hr_dir = os.path.join(opt.results_dir, 'HR_VHE')
    os.makedirs(lr_dir, exist_ok=True)
    if not opt.vs_only:
        os.makedirs(hr_dir, exist_ok=True)

    image_paths = collect_images(opt.dataroot)
    if not image_paths:
        raise FileNotFoundError('No images found under %s' % opt.dataroot)
    print('found %d images in %s' % (len(image_paths), opt.dataroot))

    netG = load_vs_model(opt, device, gpu_ids)
    netSR = None if opt.vs_only else load_sr_model(opt, device)

    start = time.time()
    for i, img_path in enumerate(image_paths):
        name = os.path.basename(img_path)
        print('[%d/%d] %s' % (i + 1, len(image_paths), name))

        inp = preprocess_vs(img_path).to(device)
        with torch.no_grad():
            fake_b = netG(inp)
        lr_bgr = tensor_to_bgr_uint8(fake_b)
        save_bgr(os.path.join(lr_dir, name), lr_bgr)
        print('  VS output %s  shape=%s' % (os.path.join(lr_dir, name), tuple(lr_bgr.shape)))

        if netSR is not None:
            sr_in = torch.from_numpy(
                np.transpose(lr_bgr[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1))
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                hr = hat_tile_inference(
                    netSR, sr_in, tile_size=opt.tile_size, tile_pad=opt.tile_pad, scale=opt.scale
                )
            hr_np = hr.squeeze(0).clamp(0, 1).detach().cpu().float().numpy()
            hr_bgr = (np.transpose(hr_np[[2, 1, 0], :, :], (1, 2, 0)) * 255.0).round().astype(np.uint8)
            save_bgr(os.path.join(hr_dir, name), hr_bgr)
            del sr_in, hr

        del inp, fake_b
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if netSR is None:
        print('done (VS only). LR_VHE -> %s  (%.2f sec)' % (lr_dir, time.time() - start))
    else:
        print('done. LR_VHE -> %s  HR_VHE -> %s  (%.2f sec)' % (lr_dir, hr_dir, time.time() - start))


if __name__ == '__main__':
    main()
