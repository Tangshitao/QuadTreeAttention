from typing import Tuple

import cv2
import numpy as np
import torch

from src.config.default import get_cfg_defaults
from src.loftr import LoFTR
from src.utils.misc import lower_config


def get_args():
    import argparse

    parser = argparse.ArgumentParser("test quadtree attention-based feature matching")
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--confidence_thresh", type=float, default=0.5)

    return parser.parse_args()


def main():
    args = get_args()

    config = get_cfg_defaults()
    config.merge_from_file(args.config_path)
    config = lower_config(config)

    matcher = LoFTR(config=config["loftr"])
    state_dict = torch.load(args.weight_path, map_location="cpu")["state_dict"]
    matcher.load_state_dict(state_dict, strict=True)

    query_image = cv2.imread(args.query_path, 1)
    ref_image = cv2.imread(args.ref_path, 1)

    new_shape = (480, 640)

    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    query_gray = cv2.resize(query_gray, new_shape[::-1])
    ref_gray = cv2.resize(ref_gray, new_shape[::-1])

    with torch.no_grad():
        batch = {
            "image0": load_torch_image(query_gray),
            "image1": load_torch_image(ref_gray),
        }

        matcher.eval()
        matcher.to("cuda")
        matcher(batch)

        query_kpts = batch["mkpts0_f"].cpu().numpy()
        ref_kpts = batch["mkpts1_f"].cpu().numpy()
        confidences = batch["mconf"].cpu().numpy()
        del batch

        conf_mask = np.where(confidences > args.confidence_thresh)
        query_kpts = query_kpts[conf_mask]
        ref_kpts = ref_kpts[conf_mask]

    def _np_to_cv2_kpts(np_kpts):
        cv2_kpts = []
        for np_kpt in np_kpts:
            cur_cv2_kpt = cv2.KeyPoint()
            cur_cv2_kpt.pt = tuple(np_kpt)
            cv2_kpts.append(cur_cv2_kpt)
        return cv2_kpts

    query_shape = query_image.shape[:2]
    ref_shape = ref_image.shape[:2]
    query_kpts = resample_kpts(
        query_kpts,
        query_shape[0] / new_shape[0],
        query_shape[1] / new_shape[1],
    )

    ref_kpts = resample_kpts(
        ref_kpts,
        ref_shape[0] / new_shape[0],
        ref_shape[1] / new_shape[1],
    )
    query_kpts, ref_kpts = _np_to_cv2_kpts(query_kpts), _np_to_cv2_kpts(ref_kpts)

    matched_image = cv2.drawMatches(
        query_image,
        query_kpts,
        ref_image,
        ref_kpts,
        [
            cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _distance=0)
            for idx in range(len(query_kpts))
        ],
        None,
        flags=2,
    )
    cv2.imwrite("result.jpg", matched_image)


def resample_kpts(kpts: np.ndarray, height_ratio, width_ratio):
    kpts[:, 0] *= width_ratio
    kpts[:, 1] *= height_ratio

    return kpts


def load_torch_image(image):
    image = torch.from_numpy(image)[None][None].cuda() / 255
    return image


if __name__ == "__main__":
    main()
