# coding: utf-8
import os
import argparse
import pathlib
import datetime
import random

import cv2
import dlib
import skimage
import imutils

import numpy as np
import pandas as pd

from collections import deque
from time import sleep, time

from detector_wrapper.enums import DetectorModel
from detector_wrapper.detector_factory import DetectorFactory
from detector_wrapper.detector_interface import DetectorInterface


def main() -> None:
    parser = argparse.ArgumentParser("Dataset preparation toolbox")
    parser.add_argument("-u", "--urls", type=pathlib.Path, default="urls/urls.txt")
    parser.add_argument("-d", "--dataset", type=pathlib.Path, default="images/dataset/")
    parser.add_argument("-ext", "--ext", type=str, default="jpg")
    parser.add_argument("-dm", "--dataset_mask", type=pathlib.Path, default="images/dataset_mask/")
    parser.add_argument("-extm", "--ext_mask", type=str, default="png")
    parser.add_argument("-iid", "--image_id", type=int, default=-1)
    parser.add_argument("-m", "--model_files", type=str, nargs="+", default=["models/model.xml"])
    parser.add_argument("-t", "--model_type", type=lambda model: DetectorModel[model], choices=list(DetectorModel), default="CvCaffe")
    parser.add_argument("-sm", "--shape_model", type=str, default="models/model.dat")
    parser.add_argument("-si", "--shape_indexes", type=int, nargs="+", default=[48, 54])
    parser.add_argument("-sam", "--shape_auto_mode", type=str, default="mean")
    parser.add_argument("-sa", "--shape_adjustment", type=float, default=1.2)
    parser.add_argument("-a", "--angle", type=float, default=0.0)
    parser.add_argument("-mc", "--min_conf", type=float, default="-inf")
    parser.add_argument("-Mc", "--max_conf", type=float, default="inf")
    parser.add_argument("-nm", "--noise_mode", type=str, default="gaussian")
    parser.add_argument("-b", "--blacklist", type=pathlib.Path, default="blacklist.txt")
    parser.add_argument("-n", "--name", type=pathlib.Path, default="dataset.csv")
    parser.add_argument("-i", "--input", type=pathlib.Path, default="images/")
    parser.add_argument("-o", "--output", type=pathlib.Path, default="images/")
    parser.add_argument("-pf", "--prefix", type=str, default="")
    parser.add_argument("-ptrain", "--perc_train", type=float, default=0.8)
    parser.add_argument("-pval", "--perc_validation", type=float, default=0.1)
    parser.add_argument("-ptest", "--perc_test", type=float, default=0.1)
    parser.add_argument("-e", "--export", action="store_true")
    parser.add_argument("-fb", "--filter_blacklist", action="store_true")
    parser.add_argument("-fc", "--filter_confidence", action="store_true")
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("-p", "--prepare", action="store_true")
    parser.add_argument("-pm", "--prepare_mask", action="store_true")
    parser.add_argument("-dt", "--detect", action="store_true")
    parser.add_argument("-ps", "--predict_shape", action="store_true")
    parser.add_argument("-an", "--add_noise", action="store_true")
    parser.add_argument("-r", "--rotate", action="store_true")
    parser.add_argument("-mr", "--mirror", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-v", "--view", action="store_true")
    parser.add_argument("-vd", "--view_detection", action="store_true")
    parser.add_argument("-vm", "--view_mask", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        df = prepare(args.dataset, args.ext)
    elif args.load:
        df = load(args.name)
        df.file_id = df.file_id.astype(str)
    if args.prepare_mask:
        df_mask = prepare(args.dataset_mask, args.ext_mask)
    if args.filter_blacklist:
        df = filter_blacklist(df, args.blacklist)
    if args.filter_confidence:
        df = filter_confidence(df, args.min_conf, args.max_conf)
    if args.rotate:
        # rotate_all(df, args.angle)
        rotate_all_random(df, args.angle)
    if args.add_noise:
        add_noise_all(df, args.noise_mode)
    if args.mirror:
        mirror_all(df)
    if args.detect:
        detector = DetectorFactory.create(args.model_type, args.model_files)
        predictor = {
            "predictor": None,
            "auto": False,
            "predict_shape": args.predict_shape,
            "shape_adjustment": args.shape_adjustment,
            "indexes": args.shape_indexes,
            "mode": args.shape_auto_mode,
        }
        try:
            predictor["predictor"] = dlib.shape_predictor(args.shape_model)
            predictor["auto"] = True
            if predictor["predict_shape"]:
                print("Shape predictor found. Automatic positioning is on.")
        except:
            if predictor["predict_shape"]:
                print("Shape predictor not found. Automatic positioning is off.")
        detect_and_export(df, args.blacklist, detector, predictor, args.output, args.prefix)
    if args.view_detection:
        detector = DetectorFactory.create(args.model_type, args.model_files)
        predictor = {
            "predictor": None,
            "auto": False,
            "predict_shape": args.predict_shape,
            "shape_adjustment": args.shape_adjustment,
            "indexes": args.shape_indexes,
            "mode": args.shape_auto_mode,
        }
        try:
            predictor["predictor"] = dlib.shape_predictor(args.shape_model)
            predictor["auto"] = True
            if predictor["predict_shape"]:
                print("Shape predictor found. Automatic positioning is on.")
        except:
            if predictor["predict_shape"]:
                print("Shape predictor not found. Automatic positioning is off.")
        view_detection(df, args.blacklist, detector, predictor)
    if args.view_mask:
        detector = DetectorFactory.create(args.model_type, args.model_files)
        predictor = {
            "predictor": None,
            "auto": False,
            "predict_shape": args.predict_shape,
            "shape_adjustment": args.shape_adjustment,
            "indexes": args.shape_indexes,
            "mode": args.shape_auto_mode,
        }
        try:
            predictor["predictor"] = dlib.shape_predictor(args.shape_model)
            predictor["auto"] = True
            print("Shape predictor found. Automatic positioning is on.")
        except:
            print("Shape predictor not found. Automatic positioning is off.")
        view_mask(df, df_mask, args.blacklist, detector, predictor, args.output, args.image_id)
    if args.view:
        view(df, args.blacklist)
    if args.save:
        save(df, args.name)
    if args.export:
        export(df, args.output, args.name)


def get_name(dataset: pathlib.Path) -> str:
    return dataset.parts[-1].split(".")[0]


def create_folder(path: pathlib.Path, name: str) -> pathlib.Path:
    folder = path / name
    folder.mkdir(exist_ok=True)
    return folder


def create_file(path: pathlib.Path, name: str) -> pathlib.Path:
    filename = path / name
    return filename


def read_file(filename: pathlib.Path) -> list:
    file_metadata = []
    file_id = filename.parts[-1].split(".")[0]
    image_file = pathlib.PurePosixPath(filename)
    # blacklist = False
    # mirror = False
    # angle = 0
    # # Box csv
    # box_id = -1
    # class = -1
    # confidence = -1
    # image_id =
    file_metadata.append([file_id, image_file])  # , blacklist, mirror, angle])
    return file_metadata


def get_data(dataset: pathlib.Path, ext='jpg') -> list:
    data = list()
    for filename in dataset.rglob(f"*.{ext}"):
        data += read_file(filename)
    return data


def prepare(dataset: pathlib.Path, ext='jpg') -> pd.DataFrame:
    data = get_data(dataset, ext)
    df = pd.DataFrame(
        data,
        columns=[
            "file_id",
            "image",
            # "blacklist",
            # "mirror",
            # "angle",
        ],
    )
    return df


def filter_confidence(
    df: pd.DataFrame, min_conf: float, max_conf: float
) -> pd.DataFrame:
    mask = df.confidence.between(min_conf, max_conf)
    return df[mask]


def filter_blacklist(df: pd.DataFrame, blacklist: pathlib.Path) -> pd.DataFrame:
    black = read_blacklist(blacklist)
    mask = ~df.file_id.isin(black)
    return df[mask]


def read_blacklist(blacklist: pathlib.Path) -> list:
    if blacklist.exists() and blacklist.is_file():
        with blacklist.open("r") as f:
            return [line.replace("\n", "") for line in f.readlines()]
    return []


def get_opt_length(center: int, length: int, max_length: int) -> tuple:
    half = int(length / 2)
    padding_1 = center - half
    padding_2 = max_length - (center + half)
    padding_total = max_length - length
    offset_opt = 0
    length_opt = length
    if padding_total < 0:
        length_opt = max_length
        offset_opt = int(max_length / 2 - center)
    elif padding_1 < 0:
        offset_opt = abs(padding_1)
    elif padding_2 < 0:
        offset_opt = padding_2
    return length_opt, offset_opt


def get_opt_dims(x_c: int, y_c: int, w_obj: int, h_obj: int, w: int, h: int) -> tuple:
    length_opt = 0
    x_off = 0
    y_off = 0
    if w_obj > h_obj:
        h_opt, y_off = get_opt_length(y_c, w_obj, h)
        length_opt, x_off = get_opt_length(x_c, h_opt, w)
    elif w_obj < h_obj:
        w_opt, x_off = get_opt_length(x_c, h_obj, w)
        length_opt, y_off = get_opt_length(y_c, w_opt, h)
    else:
        length_opt = w_obj
    return length_opt, x_off, y_off


def get_box(x_c: int, y_c: int, x_off: int, y_off: int, length_opt: int) -> tuple:
    half = int(length_opt / 2)
    p1 = (x_c - half + x_off, y_c - half + y_off)
    p2 = (x_c + half + x_off, y_c + half + y_off)
    return p1, p2


def get_square_box(
    image: np.ndarray, rect: (int, int, int, int), scale: float
) -> tuple:
    h, w, _ = image.shape
    x, y, w_obj, h_obj = rect
    x_c = x + w_obj // 2
    y_c = y + h_obj // 2
    w_obj *= scale
    h_obj *= scale
    length_opt, x_off, y_off = get_opt_dims(x_c, y_c, w_obj, h_obj, w, h)
    p1, p2 = get_box(x_c, y_c, x_off, y_off, length_opt)
    return p1, p2


def set_xywh(obj: pd.Series, p1: tuple, p2: tuple) -> pd.Series:
    obj.x1 = p1[0]
    obj.y1 = p1[1]
    obj.x2 = p2[0]
    obj.y2 = p2[1]
    obj.w = p2[0] - p1[0]
    obj.h = p2[1] - p1[1]
    obj.ratio = obj.w / obj.h
    return obj


def set_xyrb(obj: pd.Series, p1: tuple, size: tuple) -> pd.Series:
    obj.x1 = p1[0]
    obj.y1 = p1[1]
    obj.x2 = p1[0] + size[0]
    obj.y2 = p1[1] + size[1]
    obj.w = size[0]
    obj.h = size[1]
    obj.ratio = obj.w / obj.h
    return obj


def correct_aspect_ratio(df: pd.DataFrame, blacklist: pathlib.Path) -> pd.DataFrame:
    res_df = pd.DataFrame(
        columns=[
            "file_id",
            "type_of_set",
            "image",
            "annotation",
            "class_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "w",
            "h",
            "ratio",
        ]
    )
    for index, obj in df.sort_values(by=["file_id"]).iterrows():
        black = read_blacklist(blacklist)
        if obj.file_id in black:
            print("Skipping " + obj.file_id + "...")
        else:
            image = cv2.imread(str(obj.image))
            p1 = (int(obj.x1), int(obj.y1))
            p2 = (int(obj.x2), int(obj.y2))
            p1, p2 = get_square_box(image, obj)
            obj = set_xywh(obj, p1, p2)
            res_df = res_df.append(obj, ignore_index=True)
    return res_df


def create_obj_from_roi(obj: pd.Series, roi: list) -> pd.Series:
    obj = obj.copy()
    x1, y1, w_obj, h_obj = roi
    obj = set_xyrb(obj, (x1, y1), (w_obj, h_obj))
    return obj


def correct_aspect_ratio_from_rois(
    image: np.ndarray, obj: pd.Series, rois: list
) -> list:
    res = list()
    for roi in rois:
        r = create_obj_from_roi(obj, roi)
        p1, p2 = get_square_box(image, r)
        r = set_xywh(r, p1, p2)
        res.append(r)
    return res


def display(obj: pd.Series, name: str) -> None:
    image = cv2.imread(str(obj.image))
    cv2.imshow(name, image)


def postprocess_roi(image: np.ndarray, rect: [int, int, int, int]) -> (int, int, int, int, (int, int), (int, int)):
    p1, p2 = get_square_box(image, rect, scale=0.85)
    height, width = image.shape[:2]
    x1, y1 = p1
    w, h = p2[0] - x1, p2[1] - y1
    xc = x1
    yc = y1 # int(y1 + 0.2 * h)
    h = h # int(0.4 * h)
    right = xc + w
    bottom = yc + h
    if right > width:
        dw = right - width
        xc = xc - dw
    if bottom > height:
        dh = bottom - height
        yc = yc - dh
    return xc, yc, w, h, p1, p2


def get_mask_pivot(image: np.ndarray, rect: list, predictor: dict):
    x, y, w, h = rect
    x_p = x + w // 2
    y_p = y + 3 * h // 4
    pivot = (x_p, y_p)
    coords = None
    roi = [x_p - w // 2, y_p - h // 2, w, h]
    if predictor["auto"]:
        det = (x, y, x + w, y + h)
        rect = dlib.rectangle(*det)
        shape = predictor["predictor"](image, rect)
        coords = np.zeros((len(predictor["indexes"]), 2))
        for i, j in enumerate(predictor["indexes"]):
            coords[i, 0] = shape.part(j).x
            coords[i, 1] = shape.part(j).y
        if predictor["mode"] == "mean":
            mean = coords.mean(axis=0)
            pivot = (int(mean[0]), int(mean[1]))
            xc, yc = pivot
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0)
            w_s = x1 - x0
            h_s = y1 - y0
            size = int(max(w_s, h_s) *  predictor["shape_adjustment"])
            roi = [xc - size // 2, yc - size // 2, size, size]
    return pivot, coords, roi
        

def draw_mask(image: np.ndarray, mask_image: np.ndarray, mask: np.ndarray, mask_properties: dict, pivot: tuple) -> np.ndarray:
    mask_image = np.clip(mask_properties["alpha"] * mask_image + mask_properties["beta"], 0, 255).astype(np.uint8)
    h_img, w_img = image.shape[:2]
    h_mask, w_mask = mask.shape[:2]
    dx_mask = mask_properties["dx"]
    dy_mask = mask_properties["dy"]
    x_mask = mask_properties["x"]
    y_mask = mask_properties["y"]
    if x_mask == -1 or y_mask == -1:
        x_p, y_p = pivot
    else:
        x_p, y_p = x_mask, y_mask
    x_p += dx_mask
    y_p += dy_mask
    x0_roi = int(np.floor(x_p - w_mask / 2))
    x1_roi = int(np.floor(x_p + w_mask / 2))
    y0_roi = int(np.ceil(y_p - h_mask / 2))
    y1_roi = int(np.ceil(y_p + h_mask / 2))
    x0_mask = 0
    x1_mask = w_mask
    y0_mask = 0
    y1_mask = h_mask
    if x0_roi < 0:
        x0_mask = abs(x0_roi)
        x0_roi = 0
    if y0_roi < 0:
        y0_mask = abs(y0_roi)
        y0_roi = 0
    if x1_roi > w_img:
        x1_mask -= abs(w_img - x1_roi)
        x1_roi = w_img
    if y1_roi > h_img:
        y1_mask -= abs(h_img - y1_roi)
        y1_roi = h_img
    if x0_mask >= 0 and x0_mask <= w_mask and y0_mask >= 0 and y0_mask <= h_mask and x1_mask >= 0 and x1_mask <= w_mask and y1_mask >= 0 and y1_mask <= h_mask:
        roi = image[y0_roi:y1_roi, x0_roi:x1_roi]
        roi_mask = mask[y0_mask:y1_mask, x0_mask:x1_mask]
        roi_mask_image = mask_image[y0_mask:y1_mask, x0_mask:x1_mask]
        assert roi.shape == roi_mask_image.shape, f"Assertion error: roi shape {roi.shape} must be equal roi mask shape {roi_mask.shape}"
        roi[roi_mask == 255] = roi_mask_image[roi_mask == 255]
    return image


def display_objects_and_mask(obj: pd.Series, obj_mask: pd.Series, mask_properties: dict, name: str, detector: DetectorInterface, predictor: dict) -> np.ndarray:
    image = cv2.imread(str(obj.image))
    mask = cv2.imread(str(obj_mask.image), cv2.IMREAD_UNCHANGED)
    scale = mask_properties["scale"]
    mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask_image = mask[:, :, :3]
    mask = mask[:, :, 3]
    rects = []
    confs = []
    classes = []
    whitelist = [1, 502]
    class_names = {1: "face", 502: "face"}
    detector.detect(
        image=image,
        rects=rects,
        confs=confs,
        classes=classes,
        class_whitelist=whitelist,
        class_names=class_names,
    )
    detector.postprocess(
        image=image,
        rects=rects,
        confs=confs,
        nms_threshold=0.1,
        classes=classes,
        class_whitelist=whitelist,
        class_names=class_names,
    )
    h_img, w_img = image.shape[:2]
    default = False
    pivot = (w_img // 2, h_img // 2)
    rect = max(rects, default=None, key=lambda rect: rect[2] * rect[3])
    if rect is None:
        rect = [0, 0, w_img, h_img]
        default = True
    if not default:
        x, y, w, h, p1, p2 = postprocess_roi(image, rect)
        rect = [x, y, w, h]
        pivot, _, _ = get_mask_pivot(image, rect, predictor)
    image = draw_mask(image, mask_image, mask, mask_properties, pivot)
    cv2.imshow(name, image)
    return image


def display_objects(obj: pd.Series, name: str, detector: DetectorInterface, predictor: dict) -> None:
    image = cv2.imread(str(obj.image))
    rects = []
    confs = []
    classes = []
    whitelist = [1, 502]
    class_names = {1: "face", 502: "face"}
    detector.detect(
        image=image,
        rects=rects,
        confs=confs,
        classes=classes,
        class_whitelist=whitelist,
        class_names=class_names,
    )
    detector.postprocess(
        image=image,
        rects=rects,
        confs=confs,
        nms_threshold=0.1,
        classes=classes,
        class_whitelist=whitelist,
        class_names=class_names,
    )
    for rect in rects:
        x, y, w, h, p1, p2 = postprocess_roi(image, rect)
        rect = [x, y, w, h]
        if predictor["predict_shape"]:
            pivot, coords, rect = get_mask_pivot(image, rect, predictor)
            x, y, w, h = rect
        blue = (255, 0, 0)
        green = (0, 255, 0)
        red = (0, 0, 255)
        thickness = 2
        cv2.rectangle(image, p1, p2, blue, thickness)
        cv2.rectangle(image, (x, y), (x + w, y + h), green, thickness)
    cv2.imshow(name, image)


def detect_and_store_rois(obj: pd.Series, detector: DetectorInterface, predictor: dict) -> None:
    image = cv2.imread(str(obj.image))
    rects = []
    confs = []
    classes = []
    whitelist = [1, 502]
    class_names = {1: "face", 502: "face"}
    detector.detect(
        image=image,
        rects=rects,
        confs=confs,
        classes=classes,
        class_whitelist=whitelist,
        class_names=class_names,
    )
    detector.postprocess(
        image=image,
        rects=rects,
        confs=confs,
        nms_threshold=0.1,
        classes=classes,
        class_whitelist=whitelist,
        class_names=class_names,
    )
    height, width, _ = image.shape
    obj["img"] = image
    obj["pos"] = []
    obj["neg"] = []
    for rect in rects:
        x, y, w, h, _, _ = postprocess_roi(image, rect)
        rect = [x, y, w, h]
        if predictor["predict_shape"]:
            pivot, coords, rect = get_mask_pivot(image, rect, predictor)
            x, y, w, h = rect
        roi = image[y : y + h, x : x + w].copy()
        has_object = "positive" in str(obj.image.parent)
        if has_object:
            obj.pos.append(roi)
        else:
            obj.neg.append(roi)


def norm_split(ptrain: float, pval: float, ptest: float) -> (int, int, int):
    total = ptrain + pval + ptest
    ptrain = ptrain / total
    pval = pval / total
    ptest = ptest / total
    return ptrain, pval, ptest


def split(data: list, ptrain: float, pval: float, ptest: float) -> dict:
    ptrain, pval, ptest = norm_split(ptrain, pval, ptest)
    length = len(data)
    train_val_idx = round(ptrain * length)
    val_test_idx = train_val_idx + round(pval * length)
    random.seed(0)
    random.shuffle(data)
    data_dict = {
        "train": data[:train_val_idx],
        "validation": data[train_val_idx:val_test_idx],
        "test": data[val_test_idx:],
    }
    return data_dict


def detect_and_export(
    df: pd.DataFrame,
    blacklist: pathlib.Path,
    detector: DetectorInterface,
    predictor: dict,
    output: pathlib,
    prefix: str = "",
    ptrain: float = 0.8,
    pval: float = 0.1,
    ptest: float = 0.1,
):
    data = [obj for i, obj in df.sort_values(by=["image"]).iterrows()]
    data = split(data, ptrain, pval, ptest)
    for split_set, split_data in data.items():
        output_neg = output / split_set / "2"
        os.makedirs(str(output_neg), exist_ok=True)
        output_pos = output / split_set / "1"
        os.makedirs(str(output_pos), exist_ok=True)
        output_neither = output / split_set / "neither"
        os.makedirs(str(output_neither), exist_ok=True)
        for i, obj in enumerate(split_data):
            black = read_blacklist(blacklist)
            if obj.file_id in black:
                print("Skipping " + obj.file_id + "...")
                continue
            detect_and_store_rois(obj, detector, predictor)
            if not obj.pos and not obj.neg:
                filename = output_neither / f"{prefix}{i}_xxx.jpg"
                cv2.imwrite(str(filename), obj.img)
            for j, pos in enumerate(obj.pos):
                filename = output_pos / f"{prefix}{i}_{j}_pos.jpg"
                cv2.imwrite(str(filename), pos)
            for j, neg in enumerate(obj.neg):
                filename = output_neg / f"{prefix}{i}_{j}_neg.jpg"
                cv2.imwrite(str(filename), neg)
            del obj


def flip(obj: pd.Series) -> None:
    image = cv2.imread(str(obj.image))
    image = cv2.flip(image, 1)
    cv2.imwrite(str(obj.image), image)


def view(df: pd.DataFrame, blacklist: pathlib.Path) -> None:
    data = deque([obj for i, obj in df.sort_values(by=["image"]).iterrows()])
    direction = 0
    while True:
        obj = data[0]
        black = read_blacklist(blacklist)
        if obj.file_id in black:
            print("Skipping " + obj.file_id + "...")
            direction = 1 if direction == 0 else direction
        else:
            display(obj, "viewing mode")
            key = cv2.waitKey()
            if key == ord("q"):
                break
            elif key == ord("d"):
                direction = 1
            elif key == ord("a"):
                direction = -1
            elif key == ord("f"):
                flip(obj)
                direction = 0
            elif key == ord("c"):
                mode = "w"
                if blacklist.exists() and blacklist.is_file():
                    mode = "a"
                with blacklist.open(mode) as f:
                    f.write(obj.file_id + "\n")
            else:
                direction = 0
        data.rotate(direction)


def view_detection(
    df: pd.DataFrame, blacklist: pathlib.Path, detector: DetectorInterface, predictor: dict
) -> None:
    data = deque([obj for i, obj in df.sort_values(by=["image"]).iterrows()])
    direction = 0
    while True:
        obj = data[0]
        black = read_blacklist(blacklist)
        if obj.file_id in black:
            print("Skipping " + obj.file_id + "...")
            direction = 1 if direction == 0 else direction
        else:
            display_objects(obj, "detecting mode", detector, predictor)
            key = cv2.waitKey()
            if key == ord("q"):
                break
            elif key == ord("d"):
                direction = 1
            elif key == ord("a"):
                direction = -1
            elif key == ord("f"):
                flip(obj)
                direction = 0
            elif key == ord("c"):
                mode = "w"
                if blacklist.exists() and blacklist.is_file():
                    mode = "a"
                with blacklist.open(mode) as f:
                    f.write(obj.file_id + "\n")
            else:
                direction = 0
        data.rotate(direction)


def scale_mask(mask_properties: dict, key: str, scale: float = 0.05):
    if key == "t":
        mask_properties["scale"] += scale
    elif key == "g":
        mask_properties["scale"] -= scale
        if mask_properties["scale"] < scale:
            mask_properties["scale"] = scale


def set_mask_coord(event, x, y, flags, mask_properties):
    if event == cv2.EVENT_LBUTTONDOWN:
        mask_properties["x"] = x
        mask_properties["y"] = y
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        mask_properties["x"] = -1
        mask_properties["y"] = -1
        

def move_mask(mask_properties: dict, key: str, scale: int = 1):
    if key == "i":
        mask_properties["dy"] -= scale
    elif key == "k":
        mask_properties["dy"] += scale
    elif key == "l":
        mask_properties["dx"] += scale
    elif key == "j":
        mask_properties["dx"] -= scale


def contrast_mask(mask_properties: dict, key: str, scale: float = 0.05):
    if key == "y":
        mask_properties["alpha"] += scale
    elif key == "h":
        mask_properties["alpha"] -= scale


def brightness_mask(mask_properties: dict, key: str, scale: float = 1):
    if key == "m":
        mask_properties["beta"] += scale
    elif key == "n":
        mask_properties["beta"] -= scale


def get_default_mask_properties(mask_properties: dict = None) -> dict:
    if mask_properties is None:
        return {
            "dx": 0,
            "dy": 0,
            "scale": 1.0,
            "alpha": 1.0,
            "beta": 0,
            "x": -1,
            "y": -1,
        }
    mask_properties["dx"] = 0
    mask_properties["dy"] = 0
    mask_properties["scale"] = 1.0
    mask_properties["alpha"] = 1.0
    mask_properties["beta"] = 0
    mask_properties["x"] = -1
    mask_properties["y"] = -1
    return mask_properties


def save_image_and_mask(obj: pd.Series, image: np.ndarray, image_id: int, output: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    image_path = pathlib.Path(obj.image)
    image_file = output / f"{image_id}_{image_path.parts[-1]}"
    cv2.imwrite(str(image_file), image)


def view_mask(
    df: pd.DataFrame, df_mask: pd.DataFrame, blacklist: pathlib.Path, detector: DetectorInterface, predictor: dict, output: pathlib.Path, image_id: int = -1
) -> None:
    data = deque([obj for i, obj in df.sort_values(by=["image"]).iterrows()])
    data_mask = deque([obj for i, obj in df_mask.sort_values(by=["image"]).iterrows()])
    direction = 0
    direction_mask = 0
    window_name = "mask mode"
    mask_properties = get_default_mask_properties()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, set_mask_coord, mask_properties)
    while True:
        obj = data[0]
        obj_mask = data_mask[0]
        black = read_blacklist(blacklist)
        if obj.file_id in black:
            print("Skipping " + obj.file_id + "...")
            direction = 1 if direction == 0 else direction
            direction_mask = 1 if direction_mask == 0 else direction_mask
        else:
            image = display_objects_and_mask(obj, obj_mask, mask_properties, window_name, detector, predictor)
            key = cv2.waitKey()
            if key == ord("q"):
                break
            elif key == ord("d"):
                direction = 1
                direction_mask = 0
            elif key == ord("a"):
                direction = -1
                direction_mask = 0
            elif key == ord("w"):
                direction = 0
                direction_mask = 1
            elif key == ord("s"):
                direction = 0
                direction_mask = -1
            elif key == ord("f"):
                flip(obj)
                direction = 0
                direction_mask = 0
            elif key == ord("x"):
                image_id += 1
                save_image_and_mask(obj, image, image_id, output)
            elif chr(key) in "ijkl" :
                move_mask(mask_properties, chr(key))
                direction = 0
                direction_mask = 0
            elif chr(key) in "tg" :
                scale_mask(mask_properties, chr(key))
                direction = 0
                direction_mask = 0
            elif chr(key) in "yh" :
                contrast_mask(mask_properties, chr(key))
                direction = 0
                direction_mask = 0
            elif chr(key) in "nm" :
                brightness_mask(mask_properties, chr(key))
                direction = 0
                direction_mask = 0
            elif key == ord("c"):
                mode = "w"
                if blacklist.exists() and blacklist.is_file():
                    mode = "a"
                with blacklist.open(mode) as f:
                    f.write(obj.file_id + "\n")
            else:
                direction = 0
                direction_mask = 0
            if chr(key) in "wsr":
                mask_properties = get_default_mask_properties(mask_properties)
        data.rotate(direction)
        data_mask.rotate(direction_mask)


def rename(filename: str, suffix: str):
    name, ext = filename.split(".")
    return f"{name}_{suffix}.{ext}"


def rotate(obj: pd.Series, angle: float):
    image = cv2.imread(str(obj.image))
    image = imutils.rotate_bound(image, angle)
    full_path = list(obj.image.parts)
    full_path[-1] = rename(full_path[-1], f"angle_{int(angle)}")
    full_path = pathlib.Path(*full_path)
    cv2.imwrite(str(full_path), image)


def rotate_roi(obj: pd.Series, angle: float):
    image = cv2.imread(str(obj.image))
    image = imutils.rotate(image, angle)
    full_path = list(obj.image.parts)
    full_path[-1] = rename(full_path[-1], f"angle_{int(angle)}")
    full_path = pathlib.Path(*full_path)
    cv2.imwrite(str(full_path), image)


def rotate_all(df: pd.DataFrame, angle: float) -> None:
    for _, obj in df.sort_values(by=["image"]).iterrows():
        rotate(obj, angle)


def rotate_all_random(df: pd.DataFrame, angle_bound: float) -> None:
    for _, obj in df.sort_values(by=["image"]).iterrows():
        angle = 0
        while angle == 0:
            angle = random.randint(-angle_bound, +angle_bound)
        rotate_roi(obj, angle)


def add_noise(obj: pd.Series, mode: str):
    image = cv2.imread(str(obj.image))
    image = skimage.util.random_noise(image / 255.0, mode=mode)
    image = image * 255.0
    full_path = list(obj.image.parts)
    full_path[-1] = rename(full_path[-1], f"noise_{mode}")
    full_path = pathlib.Path(*full_path)
    cv2.imwrite(str(full_path), image.astype(np.uint8))


def add_noise_all(df: pd.DataFrame, mode: str) -> None:
    for _, obj in df.sort_values(by=["image"]).iterrows():
        add_noise(obj, mode)


def mirror(obj: pd.Series) -> None:
    image = cv2.imread(str(obj.image))
    image = cv2.flip(image, 1)
    full_path = list(obj.image.parts)
    full_path[-1] = rename(full_path[-1], "mirror")
    full_path = pathlib.Path(*full_path)
    cv2.imwrite(str(full_path), image)


def mirror_all(df: pd.DataFrame) -> None:
    for _, obj in df.sort_values(by=["image"]).iterrows():
        mirror(obj)


def load(filename: pathlib.Path) -> pd.DataFrame():
    return pd.read_csv(filename, index_col=0)


def save(df: pd.DataFrame, filename: pathlib.Path) -> None:
    df.to_csv(filename)


def export(df: pd.DataFrame, output: pathlib.Path, filename: pathlib.Path) -> None:
    name = get_name(filename)
    output = output / name
    output.mkdir(parents=True, exist_ok=True)
    for _, obj in df.sort_values(by=["file_id"]).iterrows():
        image_path = pathlib.Path(obj.image)
        image = cv2.imread(str(image_path))
        image_file = output / image_path.parts[-1]
        cv2.imwrite(str(image_file), image)


if __name__ == "__main__":
    main()
