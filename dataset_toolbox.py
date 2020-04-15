# coding: utf-8
import os
import argparse
import pathlib
import datetime
import random

import cv2
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
    parser.add_argument("-m", "--model_files", type=str, nargs="+", default=["models/model.xml"])
    parser.add_argument("-t", "--model_type", type=lambda model: DetectorModel[model], choices=list(DetectorModel), default="CvCaffe")
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
    parser.add_argument("-dt", "--detect", action="store_true")
    parser.add_argument("-an", "--add_noise", action="store_true")
    parser.add_argument("-r", "--rotate", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-v", "--view", action="store_true")
    parser.add_argument("-vd", "--view_detection", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        df = prepare(args.dataset, args.model_files)
    elif args.load:
        df = load(args.name)
        df.file_id = df.file_id.astype(str)
    if args.filter_blacklist:
        df = filter_blacklist(df, args.blacklist)
    if args.filter_confidence:
        df = filter_confidence(df, args.min_conf, args.max_conf)
    if args.rotate:
        # rotate_all(df, args.angle)
        rotate_all_random(df, args.angle)
    if args.add_noise:
        add_noise_all(df, args.noise_mode)
    if args.detect:
        detector = DetectorFactory.create(args.model_type, args.model_files)
        detect_and_export(df, args.blacklist, detector, args.output, args.prefix)
    if args.view_detection:
        detector = DetectorFactory.create(args.model_type, args.model_files)
        view_detection(df, args.blacklist, detector)
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
    file_metadata.append([file_id, image_file])  # , blacklist, mirror, angle])
    return file_metadata


def get_data(dataset: pathlib.Path) -> list:
    data = list()
    for filename in dataset.rglob("*.jpg"):
        data += read_file(filename)
    return data


def prepare(dataset: pathlib.Path, model: list) -> pd.DataFrame:
    data = get_data(dataset)
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
    yc = int(y1 + 0.2 * h)
    h = int(0.4 * h)
    right = xc + w
    bottom = yc + h
    if right > width:
        dw = right - width
        xc = xc - dw
    if bottom > height:
        dh = bottom - height
        yc = yc - dh
    return xc, yc, w, h, p1, p2

def display_objects(obj: pd.Series, name: str, detector: DetectorInterface) -> None:
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
        xc, yc, w, h, p1, p2 = postprocess_roi(image, rect)
        blue = (255, 0, 0)
        green = (0, 255, 0)
        red = (0, 0, 255)
        thickness = 2
        cv2.rectangle(image, p1, p2, blue, thickness)
        cv2.rectangle(image, (xc, yc), (xc + w, yc + h), green, thickness)
    cv2.imshow(name, image)


def detect_and_store_rois(obj: pd.Series, detector: DetectorInterface) -> None:
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
        xc, yc, w, h, _, _ = postprocess_roi(image, rect)
        roi = image[yc : yc + h, xc : xc + w].copy()
        has_object = 'positive' in str(obj.image.parent)
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
            detect_and_store_rois(obj, detector)
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
    df: pd.DataFrame, blacklist: pathlib.Path, detector: DetectorInterface
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
            display_objects(obj, "detecting mode", detector)
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


def rename(filename: str, suffix: str):
    name, ext = filename.split('.')
    return f"{name}_{suffix}.{ext}"


def rotate(obj: pd.Series, angle: float):
    image = cv2.imread(str(obj.image))
    image = imutils.rotate_bound(image, angle)
    full_path = list(obj.image.parts)
    full_path[-1] = rename(full_path[-1], f'angle_{int(angle)}')
    full_path = pathlib.Path(*full_path)
    cv2.imwrite(str(full_path), image)


def rotate_roi(obj: pd.Series, angle: float):
    image = cv2.imread(str(obj.image))
    image = imutils.rotate(image, angle)
    full_path = list(obj.image.parts)
    full_path[-1] = rename(full_path[-1], f'angle_{int(angle)}')
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
    full_path[-1] = rename(full_path[-1], f'noise_{mode}')
    full_path = pathlib.Path(*full_path)
    cv2.imwrite(str(full_path), image.astype(np.uint8))


def add_noise_all(df: pd.DataFrame, mode: str) -> None:
    for _, obj in df.sort_values(by=["image"]).iterrows():
        add_noise(obj, mode)


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
