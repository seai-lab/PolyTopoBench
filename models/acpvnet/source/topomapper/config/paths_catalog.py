import os
import os.path as osp


class DatasetCatalog(object):
    PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

    @staticmethod
    def get(name):
        if name in {
            'custom_acpv_train_with_latent_vertex_heatmap',
            'custom_acpv_val_with_latent_vertex_heatmap',
            'custom_acpv_test_with_latent_vertex_heatmap',
        }:
            custom_data_dir = os.environ.get('ACPV_DATA_DIR')
            if not custom_data_dir:
                raise KeyError(
                    "DatasetCatalog entry requires ACPV_DATA_DIR to be set: "
                    f"{name}"
                )
            custom_data_dir = osp.abspath(custom_data_dir)
            attrs = {
                'data_dir': custom_data_dir,
                'factory': (
                    'LatentVertexHeatmapTrainDataset'
                    if 'train' in name else 'LatentVertexHeatmapTestDataset'
                ),
            }
        else:
            raise KeyError(f"Unknown dataset entry: {name}")

        if 'train' in name:
            split = 'train'
        elif 'val' in name:
            split = 'val'
        elif 'test' in name:
            split = 'test'
        else:
            raise ValueError(f"Cannot determine split type from dataset name: {name}")
        
        args = dict(
            data_dir=(
                attrs['data_dir']
                if osp.isabs(attrs['data_dir'])
                else osp.join(DatasetCatalog.PROJECT_ROOT, attrs['data_dir'])
            ),
            split=split
        )

        if 'factory' in attrs:
            return dict(factory=attrs['factory'], args=args)

        if 'train' in name:
            return dict(factory="TrainDataset", args=args)
        if 'test' in name:
            if 'ann_file' in attrs:
                return dict(factory="TestDatasetWithAnnotations", args=args)
            else:
                return dict(factory="TestDataset", args=args)

        raise NotImplementedError()
