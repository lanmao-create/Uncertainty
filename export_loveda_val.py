"""Copy LoveDA Urban validation RGB images and export colored ground truth.

Run from Vaihingen: python export_loveda_val.py
Original images and index masks remain available for training.
"""
import argparse
import ast
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent


def validation_settings():
    """Read the existing split/palette without executing the training module."""
    tree = ast.parse((ROOT / 'utils_loveda.py').read_text(encoding='utf-8-sig'))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id == 'DATASET' and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == 'Urban'):
            values = {}
            for item in node.body:
                if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name):
                    name = item.targets[0].id
                    if name in ('test_ids', 'palette', 'LABELS'):
                        values[name] = ast.literal_eval(item.value)
            return values
    raise ValueError('Urban validation settings not found in utils_loveda.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=ROOT / 'ISPRS_dataset/loveDA/Urban')
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    settings = validation_settings()
    ids, palette = settings['test_ids'], settings['palette']
    # Validate every source before creating any exports.
    for tile in ids:
        for sub in ('images_png', 'masks_png'):
            path = data_root / sub / f'{tile}.png'
            if not path.is_file():
                raise FileNotFoundError(path)
    output = data_root / 'val_visual'
    rgb_dir, label_dir = output / 'RGB', output / 'Label'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for tile in tqdm(ids, desc='Export validation', unit='image'):
        source = data_root / 'images_png' / f'{tile}.png'
        mask_path = data_root / 'masks_png' / f'{tile}.png'
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or np.any((mask < 0) | (mask > 7)):
            raise ValueError(f'{mask_path}: expected original single-channel mask with values 0..7')
        with Image.open(source) as image:
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f'Image/mask size mismatch: {tile}')
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for index, color in palette.items():
            colored[mask == index + 1] = color
        # Raw 0 is ignored ground truth and remains black in this display only.
        Image.fromarray(colored).save(label_dir / f'{tile}.png')
        shutil.copy2(source, rgb_dir / f'{tile}.png')
    legend = ['0: ignore = (0, 0, 0)']
    legend += [f'{i + 1}: {name} = {palette[i]}' for i, name in enumerate(settings['LABELS'])]
    (output / 'legend.txt').write_text('\n'.join(legend) + '\n', encoding='utf-8')
    print(f'Exported {len(ids)} RGB images and {len(ids)} colored labels to {output}')


if __name__ == '__main__':
    main()
