"""Validation inference for HiSup on the PolyTopoBench mirrors.

Writes ``<OUTPUT_DIR>/<DATASETS.TEST[0]>.json`` (polygon predictions) and
``<OUTPUT_DIR>/<DATASETS.TEST[0]>_mask.json`` (RLE mask predictions).

Polygon records use the hisup layout expected by
``utilis/evaluate_vector_polygons.py --pred-type hisup``: ``segmentation`` is a
list of flat rings ``[exterior, hole1, ...]`` and ``image_id`` is taken from the
annotation file that was used for inference, so ids match the canonical GT.
"""
import json
import logging
import os.path as osp

import numpy as np
import torch
from pycocotools import mask as coco_mask
from skimage.measure import label, regionprops
from tqdm import tqdm

from hisup.dataset import build_test_dataset
from hisup.utils.comm import to_single_device
from tools.evaluation import boundary_eval, coco_eval, polis_eval


def poly_to_bbox(poly):
    lt_x = np.min(poly[:, 0])
    lt_y = np.min(poly[:, 1])
    w = np.max(poly[:, 0]) - lt_x
    h = np.max(poly[:, 1]) - lt_y
    return [float(lt_x), float(lt_y), float(w), float(h)]


def flatten_ring_indices(poly, ring_indices):
    if not ring_indices:
        return [poly]
    return [poly[np.asarray(index)] for index in ring_indices if len(index) >= 3]


def generate_coco_ann(polys, ring_indices_batch, scores, img_id):
    sample_ann = []
    for i, polygon in enumerate(polys):
        polygon_rings = flatten_ring_indices(polygon, ring_indices_batch[i])
        if len(polygon_rings) == 0:
            continue
        poly_points = np.concatenate(polygon_rings, axis=0)
        sample_ann.append({
            'image_id': int(img_id),
            'category_id': 100,
            'segmentation': [ring.ravel().tolist() for ring in polygon_rings],
            'bbox': poly_to_bbox(poly_points),
            'score': float(scores[i]),
        })
    return sample_ann


def generate_coco_mask(mask, img_id):
    sample_ann = []
    for prop in regionprops(label(mask > 0.50)):
        if (prop.bbox[2] - prop.bbox[0]) > 0 and (prop.bbox[3] - prop.bbox[1]) > 0:
            prop_mask = np.zeros_like(mask, dtype=np.uint8)
            prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
            score = np.ma.masked_array(mask, mask=(prop_mask != 1)).mean()
            encoded_region = coco_mask.encode(np.asfortranarray(prop_mask))
            sample_ann.append({
                'image_id': int(img_id),
                'category_id': 100,
                'segmentation': {
                    'size': encoded_region['size'],
                    'counts': encoded_region['counts'].decode(),
                },
                'score': float(score),
            })
    return sample_ann


class TestPipeline():
    def __init__(self, cfg, eval_type='coco_iou'):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.output_dir = cfg.OUTPUT_DIR
        self.dataset_name = cfg.DATASETS.TEST[0]
        self.eval_type = eval_type
        self.gt_file = ''
        self.dt_file = ''

    def test(self, model):
        self.test_on_annotated_split(model, self.dataset_name)

    def eval(self):
        logger = logging.getLogger("testing")
        logger.info('Evaluating (native) on {}'.format(self.eval_type))
        if self.eval_type == 'coco_iou':
            coco_eval(self.gt_file, self.dt_file)
        elif self.eval_type == 'boundary_iou':
            boundary_eval(self.gt_file, self.dt_file)
        elif self.eval_type == 'polis':
            polis_eval(self.gt_file, self.dt_file)

    def test_on_annotated_split(self, model, dataset_name):
        logger = logging.getLogger("testing")
        test_dataset, gt_file = build_test_dataset(self.cfg)
        logger.info('Testing on {} ({})'.format(dataset_name, gt_file))

        results = []
        mask_results = []
        for images, annotations in tqdm(test_dataset):
            with torch.no_grad():
                output, _ = model(images.to(self.device), to_single_device(annotations, self.device))
                output = to_single_device(output, 'cpu')

            batch_size = images.size(0)
            batch_ring_indices = output.get('poly_indices', [[] for _ in range(batch_size)])
            for b in range(batch_size):
                img_id = annotations[b]['image_id']
                results.extend(generate_coco_ann(output['polys_pred'][b], batch_ring_indices[b],
                                                 output['scores'][b], img_id))
                mask_results.extend(generate_coco_mask(output['mask_pred'][b], img_id))

        for suffix, payload in (('', results), ('_mask', mask_results)):
            dt_file = osp.join(self.output_dir, '{}{}.json'.format(dataset_name, suffix))
            logger.info('Writing {} predictions to {}'.format(len(payload), dt_file))
            with open(dt_file, 'w') as _out:
                json.dump(payload, _out)
            self.gt_file = gt_file
            self.dt_file = dt_file
            if len(payload) == 0:
                logger.warning('No predictions in %s; skipping native evaluation.', dt_file)
            else:
                self.eval()
