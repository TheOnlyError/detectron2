# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import random

import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from cityscapesscripts.helpers.labels import Label

from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

logger = logging.getLogger(__name__)


def load_semantic(subset):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []

    subpath = '../../'
    path = subpath + "rooms_augment_mask/"
    files = os.listdir(path)
    files_set = []
    for file in files:
        files_set.append(file.split(".")[0].split("_")[0])
    files_set = list(dict.fromkeys(files_set))
    random.shuffle(files_set)

    train_size = int(0.8 * len(files_set))
    if subset == 'train':
        files_set = files_set[:train_size]
    else:
        files_set = files_set[train_size:]
    for image_id in files_set:
        id = int(image_id)
        shp = mpimg.imread(subpath + 'rooms_augment_mask/{}.jpg'.format(id)).shape
        ret.append(
            {
                "file_name": subpath + 'rooms_augment_mask/{}.jpg'.format(id),
                "sem_seg_file_name": subpath+ 'rooms_augment_mask/{}_mask.png'.format(id),
                "height": shp[0],
                "width": shp[1],
            }
        )
    return ret


# def _cityscapes_files_to_dict(files, from_json, to_polygons):
#     """
#     Parse cityscapes annotation files to a instance segmentation dataset dict.
#
#     Args:
#         files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
#         from_json (bool): whether to read annotations from the raw json file or the png files.
#         to_polygons (bool): whether to represent the segmentation as polygons
#             (COCO's format) instead of masks (cityscapes's format).
#
#     Returns:
#         A dict in Detectron2 Dataset format.
#     """
#     from cityscapesscripts.helpers.labels import id2label, name2label
#
#     image_file, instance_id_file, _, json_file = files
#
#     annos = []
#
#     if from_json:
#         from shapely.geometry import MultiPolygon, Polygon
#
#         with PathManager.open(json_file, "r") as f:
#             jsonobj = json.load(f)
#         ret = {
#             "file_name": image_file,
#             "image_id": os.path.basename(image_file),
#             "height": jsonobj["imgHeight"],
#             "width": jsonobj["imgWidth"],
#         }
#
#         # `polygons_union` contains the union of all valid polygons.
#         polygons_union = Polygon()
#
#         # CityscapesScripts draw the polygons in sequential order
#         # and each polygon *overwrites* existing ones. See
#         # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
#         # We use reverse order, and each polygon *avoids* early ones.
#         # This will resolve the ploygon overlaps in the same way as CityscapesScripts.
#         for obj in jsonobj["objects"][::-1]:
#             if "deleted" in obj:  # cityscapes data format specific
#                 continue
#             label_name = obj["label"]
#
#             try:
#                 label = name2label[label_name]
#             except KeyError:
#                 if label_name.endswith("group"):  # crowd area
#                     label = name2label[label_name[: -len("group")]]
#                 else:
#                     raise
#             if label.id < 0:  # cityscapes data format
#                 continue
#
#             # Cityscapes's raw annotations uses integer coordinates
#             # Therefore +0.5 here
#             poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
#             # CityscapesScript uses PIL.ImageDraw.polygon to rasterize
#             # polygons for evaluation. This function operates in integer space
#             # and draws each pixel whose center falls into the polygon.
#             # Therefore it draws a polygon which is 0.5 "fatter" in expectation.
#             # We therefore dilate the input polygon by 0.5 as our input.
#             poly = Polygon(poly_coord).buffer(0.5, resolution=4)
#
#             if not label.hasInstances or label.ignoreInEval:
#                 # even if we won't store the polygon it still contributes to overlaps resolution
#                 polygons_union = polygons_union.union(poly)
#                 continue
#
#             # Take non-overlapping part of the polygon
#             poly_wo_overlaps = poly.difference(polygons_union)
#             if poly_wo_overlaps.is_empty:
#                 continue
#             polygons_union = polygons_union.union(poly)
#
#             anno = {}
#             anno["iscrowd"] = label_name.endswith("group")
#             anno["category_id"] = label.id
#
#             if isinstance(poly_wo_overlaps, Polygon):
#                 poly_list = [poly_wo_overlaps]
#             elif isinstance(poly_wo_overlaps, MultiPolygon):
#                 poly_list = poly_wo_overlaps.geoms
#             else:
#                 raise NotImplementedError("Unknown geometric structure {}".format(poly_wo_overlaps))
#
#             poly_coord = []
#             for poly_el in poly_list:
#                 # COCO API can work only with exterior boundaries now, hence we store only them.
#                 # TODO: store both exterior and interior boundaries once other parts of the
#                 # codebase support holes in polygons.
#                 poly_coord.append(list(chain(*poly_el.exterior.coords)))
#             anno["segmentation"] = poly_coord
#             (xmin, ymin, xmax, ymax) = poly_wo_overlaps.bounds
#
#             anno["bbox"] = (xmin, ymin, xmax, ymax)
#             anno["bbox_mode"] = BoxMode.XYXY_ABS
#
#             annos.append(anno)
#     else:
#         # See also the official annotation parsing scripts at
#         # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
#         with PathManager.open(instance_id_file, "rb") as f:
#             inst_image = np.asarray(Image.open(f), order="F")
#         # ids < 24 are stuff labels (filtering them first is about 5% faster)
#         flattened_ids = np.unique(inst_image[inst_image >= 24])
#
#         ret = {
#             "file_name": image_file,
#             "image_id": os.path.basename(image_file),
#             "height": inst_image.shape[0],
#             "width": inst_image.shape[1],
#         }
#
#         for instance_id in flattened_ids:
#             # For non-crowd annotations, instance_id // 1000 is the label_id
#             # Crowd annotations have <1000 instance ids
#             label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
#             label = id2label[label_id]
#             if not label.hasInstances or label.ignoreInEval:
#                 continue
#
#             anno = {}
#             anno["iscrowd"] = instance_id < 1000
#             anno["category_id"] = label.id
#
#             mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")
#
#             inds = np.nonzero(mask)
#             ymin, ymax = inds[0].min(), inds[0].max()
#             xmin, xmax = inds[1].min(), inds[1].max()
#             anno["bbox"] = (xmin, ymin, xmax, ymax)
#             if xmax <= xmin or ymax <= ymin:
#                 continue
#             anno["bbox_mode"] = BoxMode.XYXY_ABS
#             if to_polygons:
#                 # This conversion comes from D4809743 and D5171122,
#                 # when Mask-RCNN was first developed.
#                 contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
#                     -2
#                 ]
#                 polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
#                 # opencv's can produce invalid polygons
#                 if len(polygons) == 0:
#                     continue
#                 anno["segmentation"] = polygons
#             else:
#                 anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]
#             annos.append(anno)
#     ret["annotations"] = annos
#     return ret


if __name__ == "__main__":
    """
    Test the cityscapes dataset loader.

    Usage:
        python -m detectron2.data.datasets.cityscapes \
            cityscapes/leftImg8bit/train cityscapes/gtFine/train
    """
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data.catalog import Metadata

    logger = setup_logger(name=__name__)

    dirname = "cityscapes-data-vis"
    os.makedirs(dirname, exist_ok=True)

    dicts = load_semantic()
    logger.info("Done loading {} samples.".format(len(dicts)))

    stuff_classes = ['background', 'opening', 'wall']
    stuff_colors = [(0, 0, 0), (128, 128, 128), (255, 255, 255)]
    meta = Metadata().set(stuff_classes=stuff_classes, stuff_colors=stuff_colors)

    for d in dicts:
        img = np.array(Image.open(PathManager.open(d["file_name"], "rb")))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("a", vis.get_image()[:, :, ::-1])
        cv2.waitKey()
        # fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        # vis.save(fpath)
