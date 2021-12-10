import mmcv
import matplotlib.pyplot as plt

import copy
import os.path as osp

import mmcv
import numpy as np
from mmdet.apis import set_random_seed
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import inference_detector, show_result_pyplot
from mmcv.runner import get_dist_info, init_dist
import os
from glob import glob
from tqdm.notebook import tqdm

import sys
sys.path.append('/nfs3/personal/cmpark/project/dacon/mmdetection/')

class Endodet():        
    def train_epoch(self,cfg):

        # Build dataset
        datasets = [build_dataset(cfg.data.train)]

        # Build the detector
        model = build_detector(
            cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        # Add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES

        # Create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        train_detector(model, datasets, cfg, distributed=False, validate=True)

    def test(self):

        test_img = os.path.join(base_path, "test_img")

        test_file = glob(test_img+"/*.jpg")


        results = {
            'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
            'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[]
        }

        score_threshold = 0.4 # 0.8, 0.3, cfg.model.test_cfg.rcnn.score_thr
        
        for index, img_path in tqdm(enumerate(test_file), total = len(test_file)):
            
            file_name = img_path.split("/")[-1].split(".")[0]+".json"

            img = mmcv.imread(img_path)
            predictions = inference_detector(model, img)
            boxes, scores, labels = (list(), list(), list())

            for k, cls_result in enumerate(predictions):
                # print("cls_result", cls_result)
                if cls_result.size != 0:
                    if len(labels)==0:
                        boxes = np.array(cls_result[:, :4])
                        scores = np.array(cls_result[:, 4])
                        labels = np.array([k+1]*len(cls_result[:, 4]))
                    else:    
                        boxes = np.concatenate((boxes, np.array(cls_result[:, :4])))
                        scores = np.concatenate((scores, np.array(cls_result[:, 4])))
                        labels = np.concatenate((labels, [k+1]*len(cls_result[:, 4])))

            if len(labels) != 0:
                indexes = np.where(scores > score_threshold)
                # print(indexes)
                boxes = boxes[indexes]
                scores = scores[indexes]
                labels = labels[indexes]

                for label, score, bbox in zip(labels, scores, boxes):
                    x_min, y_min, x_max, y_max = bbox.astype(np.int64)

                    results['file_name'].append(file_name)
                    results['class_id'].append(label)
                    results['confidence'].append(score)
                    results['point1_x'].append(x_min)
                    results['point1_y'].append(y_min)
                    results['point2_x'].append(x_max)
                    results['point2_y'].append(y_min)
                    results['point3_x'].append(x_max)
                    results['point3_y'].append(y_max)
                    results['point4_x'].append(x_min)
                    results['point4_y'].append(y_max)
        return results

if __name__ == '__main__':
    config = '/nfs3/personal/cmpark/project/dacon/mmdetection/configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py'

    cfg = Config.fromfile(config)

    base_path = "/tmp" # base_dir과 같습니다.

    test_anno = "mmdet_fold/valid_0_label.json" # 출력은 "valid_partial_annotations.json"
    train_anno = "mmdet_fold/train_0_label.json" # 출력은 "train_partial_annotations.json"

    test_img =  "train_data/images" # 출력은 "train_100000"
    train_img = "train_data/images" # 출력은 "train_100000"


    save_path_dir = os.path.join(base_path, "work_dir")
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)

    # Modify dataset type and path
    cfg.dataset_type = 'CocoDataset'
    cfg.data_root = base_path

    cfg.data.test.type = 'CocoDataset'
    cfg.data.test.data_root = base_path
    cfg.data.test.ann_file = test_anno
    cfg.data.test.img_prefix = test_img

    cfg.data.train.type = 'CocoDataset'
    cfg.data.train.data_root = base_path
    cfg.data.train.ann_file = train_anno
    cfg.data.train.img_prefix = train_img

    cfg.data.val.type = 'CocoDataset'
    cfg.data.val.data_root = base_path
    cfg.data.val.ann_file = test_anno
    cfg.data.val.img_prefix = test_img

    cfg.data.samples_per_gpu = 12
    cfg.data.workers_per_gpu = 4

    classes = ('01_ulcer', '02_mass', '04_lymph', '05_bleeding')

    cfg.data.train.classes = classes
    cfg.data.val.classes = classes
    cfg.data.test.classes = classes


    # modify num classes of the model in box head
    cfg.model.bbox_head.num_classes = 4
    
    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = "/nfs3/personal/cmpark/project/dacon/mmdetection/checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth" # Error가 날 경우, "/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

    # Set up working dir to save files and logs.
    cfg.work_dir = save_path_dir

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None

    # 에폭 수 조절
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=200)

    cfg.log_config.interval = 3

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 4
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 4

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    
    distributed = True
    init_dist('pytorch', **cfg.dist_params)
    # re-set gpu_ids with distributed training mode
    _, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    mmdet_model = Endodet()
    mmdet_model.train_epoch(cfg)
