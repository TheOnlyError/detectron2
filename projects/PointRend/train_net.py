#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import sys
import time

from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, DatasetCatalog, \
    build_detection_test_loader
from floorplans import load_semantic
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
)
import matplotlib.image as mpimg
import numpy as np
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.COLOR_AUG_SSD:
        augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_instance":
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            return CityscapesSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    path = "../../output3"

    resources = False
    if resources:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=path).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        model.eval()
        single = mpimg.imread('../../resources/single.jpg')
        multi = mpimg.imread('../../resources/multi.jpg')
        # image = mpimg.imread('resources/multi_large.jpg')
        # image = mpimg.imread('resources/multi_largest.jpg')
        # m_sampled = mpimg.imread('../../resources/m_sampled.jpg')
        # m_sampled2 = mpimg.imread('../../resources/m_sampled2.jpg')
        # mplan_s = mpimg.imread('../../resources/mplan_s.jpg')
        images = [single, multi]
        size = 512
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size))
        ])
        for i, image in enumerate(images):
            image = trans(image)
            inputs = [{
                'image': image,
                'sem_seg': trans(np.random.random((size, size)))
            }]

            result = model(inputs)[0]['sem_seg'].cpu().detach().numpy()
            result = np.moveaxis(result, 0, -1)
            result = result.argmax(axis=-1)
            result[result == 1] = 10
            result[result == 0] = 20
            result[result == 2] = 30

            timestr = time.strftime("%Y%m%d-%H%M%S")
            mpimg.imsave("result" + timestr + str(i) + "_pointrend.jpg", result.astype(np.uint8))
        return

    stuff_classes = ['bg', 'opening', 'wall']
    stuff_colors = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    DatasetCatalog.register(
        "floorplans_sem_seg_train", lambda subset='train': load_semantic(subset)
    )
    MetadataCatalog.get("floorplans_sem_seg_train").set(evaluator_type="sem_seg", stuff_classes=stuff_classes,
                                                        stuff_colors=stuff_colors)
    DatasetCatalog.register(
        "floorplans_sem_seg_val", lambda subset='val': load_semantic(subset)
    )
    MetadataCatalog.get("floorplans_sem_seg_val").set(evaluator_type="sem_seg", stuff_classes=stuff_classes,
                                                      stuff_colors=stuff_colors)

    predict = True
    if predict:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=path).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )

        model.eval()
        data_loader = build_detection_test_loader(cfg, "floorplans_sem_seg_train")
        samples = 2
        i = 0
        for idx, inputs in enumerate(data_loader):
            result = model(inputs)[0]['sem_seg'].cpu().detach().numpy()
            result = np.moveaxis(result, 0, -1)
            result = result.argmax(axis=-1)
            result[result == 1] = 10
            result[result == 0] = 20
            result[result == 2] = 30

            timestr = time.strftime("%Y%m%d-%H%M%S")
            mpimg.imsave("result" + timestr + str(i) + "_input_pointrend.jpg",
                         np.moveaxis(inputs[0]["image"].cpu().detach().numpy(), 0, -1))
            mpimg.imsave("result" + timestr + str(i) + "_mask_pointrend.jpg", inputs[0]["sem_seg"])
            mpimg.imsave("result" + timestr + str(i) + "_pointrend.jpg", result.astype(np.uint8))

            i += 1
            if i == samples:
                break
        return

        # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        #     cfg.MODEL.WEIGHTS, resume=args.resume
        # )
        # res = Trainer.test(cfg, model)
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        # return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        1,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
