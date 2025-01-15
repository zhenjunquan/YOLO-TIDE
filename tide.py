import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default=r'val_coco_format.json', help='training model path')
    parser.add_argument('--pred_json', type=str, default=r'predictions.json', help='data yaml path')

    return parser.parse_known_args()[0]




if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    # pred_json = cover_pred_json_bbox(anno_json, pred_json)

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    tide = TIDE()
    tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    tide.summarize()
    tide.plot(out_dir='TIDE-result')
# 报错的时候 设置 
		# mask   = f.toRLE(ann['segmentation'], image_lookup[image]['width'], image_lookup[image]['height'])

		# mask = None


