import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))
    
    DATASETS = {}

    @staticmethod
    def get(name):
        data_dir = os.environ.get("POLYTOPOBENCH_DATA_PROCESSED_ROOT", DatasetCatalog.DATA_DIR)
        if name in {
            'custom_hisup_train',
            'custom_hisup_val',
            'custom_hisup_test',
        }:
            custom_root = os.environ.get("HISUP_DATA_ROOT")
            if not custom_root:
                raise KeyError(
                    "DatasetCatalog entry requires HISUP_DATA_ROOT to be set: "
                    f"{name}"
                )
            custom_root = osp.abspath(custom_root)
            split = "train" if "train" in name else ("val" if "val" in name else "test")
            attrs = {
                'img_dir': os.path.join(custom_root, split, 'images'),
                'ann_file': os.path.join(custom_root, split, 'annotation.json'),
            }
        elif name in DatasetCatalog.DATASETS:
            attrs = DatasetCatalog.DATASETS[name]
        else:
            raise KeyError(f"Unknown dataset entry: {name}")

        root = attrs['img_dir']
        ann_file = attrs['ann_file']
        if not osp.isabs(root):
            root = osp.join(data_dir, root)
        if not osp.isabs(ann_file):
            ann_file = osp.join(data_dir, ann_file)

        args = dict(
            root=root,
            ann_file=ann_file
        )

        if 'train' in name:
            return dict(factory="TrainDataset", args=args)
        if 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations", args=args)
        raise NotImplementedError()
