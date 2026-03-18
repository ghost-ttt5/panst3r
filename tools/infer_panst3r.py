#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from must3r.datasets import ImgNorm
from must3r.model import get_pointmaps_activation
from must3r.tools.image import get_resize_function, is_valid_pil_image_file
from must3r.engine.inference import postprocess
from must3r.model.blocks.attention import toggle_memory_efficient_attention, has_xformers
from dust3r.viz import rgb

from panst3r import PanSt3R
from panst3r.class_names import CLASS_NAMES
from panst3r.datasets import id2rgb, rgb2id
from panst3r.engine import panoptic_inference_qubo, panoptic_inference_v1, panoptic_inference_v2
from panst3r.utils import get_colors_grid


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Headless PanSt3R inference script (no gradio frontend).')
    parser.add_argument('--weights', type=str, required=True, help='Path to PanSt3R model weights.')
    parser.add_argument('--retrieval', type=str, default=None, help='Path to retrieval weights (optional).')
    parser.add_argument('--input', type=str, required=True, help='Input image folder or a text file with image paths.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--image_size', type=int, default=512, choices=[512, 384, 224, 336, 448, 768], help='Input image size.')
    parser.add_argument('--num_keyframes', type=int, default=None, help='Number of keyframes. Defaults to all images.')
    parser.add_argument('--class_set', nargs='+', default=['scannet'], choices=sorted(CLASS_NAMES.keys()),
                        help='Class presets for open-vocabulary inference.')
    parser.add_argument('--postprocess_fn', choices=['qubo', 'standard_v1', 'standard_v2'], default='standard_v2')
    parser.add_argument('--use_retrieval', action='store_true', help='Use retrieval-based keyframe selection.')
    parser.add_argument('--device', type=str, default='cuda', help='PyTorch device.')
    parser.add_argument('--amp', choices=['False', 'bf16', 'fp16'], default='False',
                        help='Automatic mixed precision mode.')
    parser.add_argument('--scene_conf_thr', type=float, default=3.0,
                        help='Confidence threshold used for exported scene point cloud.')
    parser.add_argument('--no_export_scene', action='store_true',
                        help='Disable exporting merged scene point cloud PLY.')
    parser.add_argument('--verbose', action='store_true')
    return parser


def load_images(paths: list[Path], size: int, patch_size: int, verbose: bool = True):
    imgs = []
    for path in paths:
        rgb_image = PIL.Image.open(path).convert('RGB')
        rgb_image.load()
        width, height = rgb_image.size
        resize_func, _, _ = get_resize_function(size, patch_size, height, width)
        rgb_tensor = resize_func(ImgNorm(rgb_image))
        imgs.append(dict(img=rgb_tensor, true_shape=np.int32([rgb_tensor.shape[-2], rgb_tensor.shape[-1]])))
        if verbose:
            print(f' - adding {path} with resolution {width}x{height} -> {rgb_tensor.shape[-1]}x{rgb_tensor.shape[-2]}')

    if len(imgs) == 1:
        imgs = imgs * 2
    return imgs


def prepare_panoptic_visualization(pan_preds: dict):
    pan_masks = [pan.cpu().detach().numpy() for pan in pan_preds['pan']]
    colors = get_colors_grid(len(pan_preds['segments_info']))
    id2color = {seg['id']: rgb2id(color) for seg, color in zip(pan_preds['segments_info'], colors)}

    pan_vis = [np.zeros_like(pan) for pan in pan_masks]
    for seg in pan_preds['segments_info']:
        color = id2color[seg['id']]
        for pan_vis_i, pan_mask_i in zip(pan_vis, pan_masks):
            pan_vis_i[pan_mask_i == seg['id']] = color
    pan_vis = [id2rgb(pan).astype(np.uint8) for pan in pan_vis]
    return pan_masks, pan_vis


def resolve_input(input_value: str) -> list[Path]:
    input_path = Path(input_value)
    if input_path.is_dir():
        image_paths = sorted([p for p in input_path.iterdir() if p.is_file() and is_valid_pil_image_file(str(p))])
    elif input_path.is_file():
        image_paths = [Path(p.strip()) for p in input_path.read_text().splitlines() if p.strip()]
    else:
        raise ValueError(f'Input path does not exist: {input_value}')

    if len(image_paths) == 0:
        raise ValueError('No valid input images found.')
    return image_paths


def save_scene_pointcloud_ply(
    output_path: Path,
    x_out: list[dict],
    imgs: list[torch.Tensor],
    true_shape: torch.Tensor,
    min_conf_thr: float = 3.0,
) -> int:
    points_all = []
    colors_all = []

    for i, pred in enumerate(x_out):
        pts3d = pred['pts3d'].cpu().numpy().reshape(-1, 3)
        conf = pred['conf'].cpu().numpy().reshape(-1)

        rgb_i = rgb(imgs[i].cpu(), true_shape[i].cpu()).reshape(-1, 3).astype(np.uint8)
        valid = np.isfinite(pts3d).all(axis=-1) & np.isfinite(conf) & (conf >= min_conf_thr)

        if np.any(valid):
            points_all.append(pts3d[valid])
            colors_all.append(rgb_i[valid])

    if len(points_all) == 0:
        raise ValueError('No valid points remained after confidence filtering; scene point cloud was not exported.')

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)

    with output_path.open('w', encoding='utf-8') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(points, colors):
            f.write(f'{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n')

    return int(points.shape[0])


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    toggle_memory_efficient_attention(enabled=has_xformers)

    image_paths = resolve_input(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    amp = False if args.amp == 'False' else args.amp

    model = PanSt3R.from_checkpoint(args.weights, args.retrieval)
    model.verbose = args.verbose
    model.panoptic_decoder.text_encoder.change_mode(fixed_vocab=False)
    model = model.eval().to(args.device)

    views = load_images(image_paths, size=args.image_size, patch_size=model.must3r_encoder.patch_size, verbose=args.verbose)
    imgs = [b['img'].to(args.device) for b in views]
    true_shape = torch.stack([torch.from_numpy(b['true_shape']) for b in views], dim=0).to(args.device)

    all_classes = sorted({c for class_set in args.class_set for c in CLASS_NAMES[class_set]})
    num_keyframes = args.num_keyframes
    if num_keyframes is None:
        num_keyframes = len(imgs)
    num_keyframes = max(2, min(num_keyframes, len(imgs)))

    out_3d, pan_out = model.forward_inference_multi_ar(
        imgs,
        true_shape,
        all_classes,
        num_keyframes=num_keyframes,
        use_retrieval=args.use_retrieval,
        max_bs=1,
        outdevice='cpu',
        amp=amp,
    )

    size = true_shape.cpu().numpy()
    label_mode = model.panoptic_decoder.label_mode
    if args.postprocess_fn == 'qubo':
        pan_preds = panoptic_inference_qubo(pan_out['pred_logits'], pan_out['pred_masks'], size, label_mode=label_mode, device='cpu', multi_ar=True)
    elif args.postprocess_fn == 'standard_v1':
        pan_preds = panoptic_inference_v1(pan_out['pred_logits'], pan_out['pred_masks'], size, label_mode=label_mode, device='cpu', multi_ar=True)
    else:
        pan_preds = panoptic_inference_v2(pan_out['pred_logits'], pan_out['pred_masks'], size, label_mode=label_mode, device='cpu', multi_ar=True)

    pointmaps_activation = get_pointmaps_activation(model.must3r_decoder)
    x_out = [postprocess(pmi[0], pointmaps_activation=pointmaps_activation) for pmi in out_3d]

    preds = pan_preds[0]
    for seg in preds['segments_info']:
        if 'category_name' not in seg:
            seg['category_name'] = all_classes[seg['category_id']]

    pan_masks, pan_vis = prepare_panoptic_visualization(preds)

    (output_dir / 'segments_info.json').write_text(json.dumps(preds['segments_info'], indent=2, ensure_ascii=False))
    (output_dir / 'inference_config.json').write_text(json.dumps({
        'input': [str(p) for p in image_paths],
        'weights': args.weights,
        'retrieval': args.retrieval,
        'image_size': args.image_size,
        'class_set': args.class_set,
        'num_keyframes': num_keyframes,
        'postprocess_fn': args.postprocess_fn,
        'use_retrieval': args.use_retrieval,
        'amp': amp,
        'device': args.device,
        'scene_conf_thr': args.scene_conf_thr,
        'export_scene': not args.no_export_scene,
    }, indent=2, ensure_ascii=False))

    for idx, image_path in enumerate(image_paths):
        stem = image_path.stem
        np.save(output_dir / f'{idx:04d}_{stem}_panoptic_ids.npy', pan_masks[idx])
        PIL.Image.fromarray(pan_vis[idx]).save(output_dir / f'{idx:04d}_{stem}_panoptic_vis.png')

        np.savez_compressed(
            output_dir / f'{idx:04d}_{stem}_geometry.npz',
            pts3d=x_out[idx]['pts3d'].cpu().numpy(),
            pts3d_local=x_out[idx]['pts3d_local'].cpu().numpy(),
            conf=x_out[idx]['conf'].cpu().numpy(),
        )

    if not args.no_export_scene:
        num_points = save_scene_pointcloud_ply(
            output_path=output_dir / 'scene_pointcloud.ply',
            x_out=x_out,
            imgs=imgs,
            true_shape=true_shape,
            min_conf_thr=args.scene_conf_thr,
        )
        print(f'Exported merged scene point cloud with {num_points} points: {output_dir / "scene_pointcloud.ply"}')

    print(f'Inference completed. Results saved to: {output_dir}')


if __name__ == '__main__':
    parser = get_args_parser()
    main(parser.parse_args())
