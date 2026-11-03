"""Shared settings, RGB/TOP patch loading and metrics for ISPRS and LoveDA."""
from pathlib import Path
import ast
from functools import lru_cache
import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
from skimage import io
import tifffile
import os

ROOT = Path(__file__).resolve().parent
PRETRAIN_FILES = {
    'ABCNet': ROOT / 'pretrain/resnet18.a1_in1k.pth',
    'CMTFNet': ROOT / 'pretrain/resnet50-0676ba61.pth',
    'UNetformer': ROOT / 'pretrain/resnet18.fb_swsl_ig1b_ft_in1k.pth',
    'DeepKANSeg': ROOT / 'pretrain/resnet18.fb_swsl_ig1b_ft_in1k.pth',
    'FTUNetformer': ROOT / 'pretrain/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth',
}
# Edit these defaults, or override them with train.py command-line arguments.
# MODEL = 'UNetformer'
# MODEL = 'FTUNetformer'
MODEL = 'ABCNet'
# MODEL = 'CMTFNet'
# MODEL = 'DeepKANSeg'  # Official default: ResNet-18 + DeepKAN + GLKAN decoder.
MODE = 'Train'
DATASET = 'Vaihingen'  # 'Vaihingen', 'Potsdam', or 'LoveDA' (the supplied Urban split)
LOSS = 'SEG'
WINDOW_SIZE = (256, 256)
STRIDE = 32
BATCH_SIZE = 10
EPOCHS = 50
STEPS_PER_EPOCH = 1000
BASE_LR = 0.01
LR_MILESTONES = (25, 35, 45)
VAL_EVERY_STEPS = 0  # 0: epoch-end validation; e.g. 500: extra validation every 500 updates.
CACHE = True
IN_CHANNELS = 3
IGNORE_INDEX = 255
LABELS = ['roads', 'buildings', 'low veg.', 'trees', 'cars', 'clutter']
N_CLASSES = len(LABELS)
WEIGHTS = torch.ones(N_CLASSES)
FOLDER = str(ROOT / 'ISPRS_dataset') + os.sep
palette = {0: (255,255,255), 1: (0,0,255), 2: (0,255,255),
           3: (0,255,0), 4: (255,255,0), 5: (255,0,0), 6: (0,0,0)}
invert_palette = {v:k for k,v in palette.items()}
_ISPRS_LABELS = tuple(LABELS)
_ISPRS_PALETTE = dict(palette)
MEAN_CLASS_COUNT = 5
LABEL_ENCODING = 'isprs_color'


@lru_cache(maxsize=1)
def loveda_settings():
    """Read the supplied Urban constants verbatim without importing/editing its code."""
    source = Path(__file__).resolve().with_name('utils_loveda.py')
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    for node in ast.walk(tree):
        condition = node.test if isinstance(node, ast.If) else None
        if not (isinstance(condition, ast.Compare)
                and isinstance(condition.left, ast.Name) and condition.left.id == 'DATASET'
                and len(condition.ops) == 1 and isinstance(condition.ops[0], ast.Eq)
                and isinstance(condition.comparators[0], ast.Constant)
                and condition.comparators[0].value == 'Urban'):
            continue
        settings = {}
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or not isinstance(statement.targets[0], ast.Name):
                continue
            name, value = statement.targets[0].id, statement.value
            if name in ('train_ids', 'test_ids', 'LABELS', 'palette', 'Stride_Size'):
                settings[name] = ast.literal_eval(value)
            elif name.endswith('_FOLDER') and isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                # Preserve relative suffixes such as images_png/{}.png; the
                # source's machine-specific Linux root is supplied at runtime.
                if isinstance(value.right, ast.Constant) and isinstance(value.right.value, str):
                    settings[name] = value.right.value
        required = ('train_ids','test_ids','LABELS','palette','Stride_Size',
                    'MAIN_FOLDER','DATA_FOLDER','LABEL_FOLDER','ERODED_FOLDER')
        if any(key not in settings for key in required):
            raise ValueError(f'Incomplete Urban settings in {source}')
        if set(settings['train_ids']) & set(settings['test_ids']):
            raise ValueError('The supplied LoveDA training and validation lists overlap')
        return settings
    raise ValueError(f'No DATASET == Urban settings in {source}')


def configure_dataset(name=None, data_root=None):
    """Set dataset-specific file templates; paths default to this project."""
    global DATASET, MAIN_FOLDER, DATA_FOLDER, LABEL_FOLDER, ERODED_FOLDER
    global BOUNDARY_FOLDER, OBJECT_FOLDER, train_ids, test_ids, Stride_Size
    global LABELS, N_CLASSES, WEIGHTS, palette, invert_palette, MEAN_CLASS_COUNT, LABEL_ENCODING
    DATASET = name or DATASET
    if DATASET in ('LoveDA', 'Urban'):
        DATASET = 'LoveDA'
    LABELS, palette = list(_ISPRS_LABELS), dict(_ISPRS_PALETTE)
    MEAN_CLASS_COUNT, LABEL_ENCODING = 5, 'isprs_color'
    default_root = ROOT / 'ISPRS_dataset' / DATASET
    if DATASET == 'LoveDA':
        default_root = ROOT / 'ISPRS_dataset' / 'loveDA' / loveda_settings()['MAIN_FOLDER']
    root = Path(data_root).expanduser().resolve() if data_root else default_root
    MAIN_FOLDER = str(root) + os.sep
    Stride_Size = STRIDE
    if DATASET == 'Vaihingen':
        train_ids = ['1','3','23','26','7','11','13','28','17','32','34','37']
        test_ids = ['5','21','15','30']
        DATA_FOLDER = str(root / 'ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area{}.tif')
        LABEL_FOLDER = str(root / 'ISPRS_semantic_labeling_Vaihingen/gts_for_participants/top_mosaic_09cm_area{}.tif')
        ERODED_FOLDER = str(root / 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE/top_mosaic_09cm_area{}_noBoundary.tif')
        BOUNDARY_FOLDER = str(root / 'top_Boundary/top_mosaic_09cm_area{}_Boundary.tif')
        OBJECT_FOLDER = str(root / 'top_objects/top_mosaic_09cm_area{}_objects.tif')
    elif DATASET == 'Potsdam':
        train_ids = ['6_10','7_10','2_12','3_11','2_10','7_8','5_10','3_12','5_12',
                     '7_11','7_9','6_9','7_7','4_12','6_8','6_12','6_7','4_11']
        test_ids = ['4_10','5_11','2_11','3_10','6_11','7_12']
        Stride_Size = 128  # User-selected Potsdam sliding-window validation stride.
        DATA_FOLDER = str(root / '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif')
        LABEL_FOLDER = str(root / '5_Labels_for_participants/top_potsdam_{}_label.tif')
        ERODED_FOLDER = str(root / '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif')
        # Potsdam SEG baseline needs no boundary/object/DSM files.
        BOUNDARY_FOLDER = OBJECT_FOLDER = None
    elif DATASET == 'LoveDA':
        settings = loveda_settings()
        train_ids, test_ids = list(settings['train_ids']), list(settings['test_ids'])
        Stride_Size = settings['Stride_Size']
        LABELS, palette = list(settings['LABELS']), dict(settings['palette'])
        MEAN_CLASS_COUNT, LABEL_ENCODING = len(LABELS), 'loveda_1_based'
        DATA_FOLDER = str(root / settings['DATA_FOLDER'])
        LABEL_FOLDER = str(root / settings['LABEL_FOLDER'])
        ERODED_FOLDER = str(root / settings['ERODED_FOLDER'])
        BOUNDARY_FOLDER = str(root / settings['BOUNDARY_FOLDER']) if 'BOUNDARY_FOLDER' in settings else None
        OBJECT_FOLDER = str(root / settings['OBJECT_FOLDER']) if 'OBJECT_FOLDER' in settings else None
    else:
        raise ValueError('Supported datasets: Vaihingen, Potsdam, LoveDA (Urban)')
    N_CLASSES, WEIGHTS = len(LABELS), torch.ones(len(LABELS))
    invert_palette = {v:k for k,v in palette.items()}
    return root


def read_image(path):
    return tifffile.imread(path) if Path(path).suffix.lower() in ('.tif', '.tiff') else io.imread(path)


def read_label(path):
    if LABEL_ENCODING == 'loveda_1_based':
        # Reading indices with Pillow also preserves masks stored in PNG mode P.
        with Image.open(path) as image:
            raw = np.asarray(image).copy()
        try:
            return decode_loveda_label(raw)
        except ValueError as exc:
            raise ValueError(
                f'Invalid LoveDA mask: {path}; shape={raw.shape}, dtype={raw.dtype}. '
                'Use original masks_png labels (single-channel indices 0..7), '
                'not RGB images from images_png or colored previews. '
                'Copy Train/Urban/masks_png and Val/Urban/masks_png into the mask directory.'
            ) from exc
    return convert_from_color(read_image(path))


def decode_loveda_label(raw):
    """Map official 0=ignore, 1..7=classes to 255=ignore, 0..6=classes."""
    if raw.ndim == 3 and raw.shape[2] == 1:
        raw = raw[..., 0]
    if (raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer)
            or np.any((raw < 0) | (raw > 7))):
        raise ValueError('LoveDA masks must be single-channel PNG label indices 0..7, not colored preview images')
    out = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)
    valid = raw > 0
    out[valid] = raw[valid] - 1
    return out


configure_dataset()


def convert_to_color(arr_2d, palette=None):
    """ Numeric labels to RGB-color encoding """
    palette = globals()['palette'] if palette is None else palette
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=None):
    """ RGB-color encoding to grayscale labels """
    palette = invert_palette if palette is None else palette
    arr_2d = np.full(arr_3d.shape[:2], IGNORE_INDEX, dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d[..., :3] == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

def object_process(object):
    ids = np.unique(object)
    new_id = 1
    for id in ids[1:]:
        object = np.where(object == id, new_id, object)
        new_id += 1
    return object

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=None, label_files=None, cache=False,
                 augmentation=True, window_size=None, steps_per_epoch=None,
                 batch_size=None, loss=None):
        super().__init__()
        self.augmentation, self.cache = augmentation, cache
        self.window_size = tuple(window_size or WINDOW_SIZE)
        self.length = (steps_per_epoch or STEPS_PER_EPOCH) * (batch_size or BATCH_SIZE)
        self.loss = loss or LOSS
        self.data_files = [(data_files or DATA_FOLDER).format(i) for i in ids]
        self.label_files = [(label_files or LABEL_FOLDER).format(i) for i in ids]
        self.boundary_files = [BOUNDARY_FOLDER.format(i) for i in ids] if 'BDY' in self.loss and BOUNDARY_FOLDER else []
        self.object_files = [OBJECT_FOLDER.format(i) for i in ids] if 'OBJ' in self.loss and OBJECT_FOLDER else []
        if ('BDY' in self.loss and not self.boundary_files) or ('OBJ' in self.loss and not self.object_files):
            raise ValueError('Boundary/object supervision requires corresponding annotations; use LOSS=SEG for Potsdam baseline')
        for f in self.data_files + self.label_files + self.boundary_files + self.object_files:
            if not Path(f).is_file():
                raise FileNotFoundError(f)
        self.data_cache_, self.label_cache_ = {}, {}
        self.boundary_cache_, self.object_cache_ = {}, {}

    def __len__(self):
        return self.length

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip = flip and random.random() < 0.5
        will_mirror = mirror and random.random() < 0.5
        result = []
        for array in arrays:
            if will_flip:
                array = np.flip(array, axis=-2)
            if will_mirror:
                array = np.flip(array, axis=-1)
            result.append(np.ascontiguousarray(array))
        return tuple(result)

    def _read(self, index, files, storage, kind):
        if index in storage:
            return storage[index]
        value = read_label(files[index]) if kind == 'label' else read_image(files[index])
        if kind == 'image':
            if value.ndim != 3 or value.shape[2] < 3:
                raise ValueError(f'Expected an HWC image with at least 3 bands: {files[index]}')
            # Potsdam RGBIR -> RGB; Vaihingen retains its native three TOP bands.
            value = value[..., :3].transpose(2, 0, 1).copy()
        elif kind == 'label':
            value[value >= N_CLASSES] = IGNORE_INDEX
        elif kind == 'boundary':
            value = (np.asarray(value) / 255).astype(np.int64)
        else:
            value = np.asarray(value, dtype=np.int64)
        if self.cache:
            storage[index] = value
        return value

    def __getitem__(self, index):
        i = random.randrange(len(self.data_files))
        image = self._read(i, self.data_files, self.data_cache_, 'image')
        label = self._read(i, self.label_files, self.label_cache_, 'label')
        if image.shape[-2:] != label.shape:
            raise ValueError(f'Image/label size mismatch: {self.data_files[i]}')
        top, bottom, left, right = get_random_pos(image, self.window_size)
        image = image[:, top:bottom, left:right].astype(np.float32) / 255.0
        label = label[top:bottom, left:right].astype(np.int64)
        boundary = self._read(i, self.boundary_files, self.boundary_cache_, 'boundary')[top:bottom, left:right] if self.boundary_files else np.zeros_like(label)
        objects = self._read(i, self.object_files, self.object_cache_, 'objects')[top:bottom, left:right] if self.object_files else np.zeros_like(label)
        arrays = (image, boundary, objects, label)
        if self.augmentation:
            arrays = self.data_augmentation(*arrays)
        image, boundary, objects, label = arrays
        if self.object_files:
            objects = object_process(objects)
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (image, boundary, objects, label))

## We load one tile from the dataset and we display it
# img = io.imread('./ISPRS_dataset/Vaihingen/top/top_mosaic_09cm_area11.tif')
# fig = plt.figure()
# fig.add_subplot(121)
# plt.imshow(img)
#
# # We load the ground truth
# gt = io.imread('./ISPRS_dataset/Vaihingen/gts_for_participants/top_mosaic_09cm_area11.tif')
# fig.add_subplot(122)
# plt.imshow(gt)
# plt.show()
#
# # We also check that we can convert the ground truth into an array format
# array_gt = convert_from_color(gt)
# print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)


# Utils

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w)
    x2 = x1 + w
    y1 = random.randint(0, H - h)
    y2 = y1 + h
    return x1, x2, y1, y2

class CrossEntropy2d_ignore(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))
        n, c, h, w = predict.size()
        valid = (target >= 0) & (target < c) & (target != self.ignore_label)
        if not valid.any():
            return predict.sum() * 0.0
        logits = predict.permute(0, 2, 3, 1)[valid]
        if weight is not None:
            weight = weight.to(device=predict.device, dtype=predict.dtype)
        return F.cross_entropy(logits, target[valid], weight=weight,
                               reduction='mean' if self.size_average else 'sum')


def loss_calc(pred, label, weights):
    return CrossEntropy2d_ignore()(pred, label.to(pred.device).long(), weights)


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

class ObjectLoss(nn.Module):
  def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

  def forward(self, pred, gt):
    num_object = int(torch.max(gt)) + 1
    num_object = min(num_object, self.max_object)
    total_object_loss = 0

    for object_index in range(1,num_object):
        mask = torch.where(gt == object_index, 1, 0).unsqueeze(1).to('cuda')
        num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to('cuda')
        avg_pool = mask / (num_point + 1)

        object_feature = pred.mul(avg_pool)

        avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(1,1,gt.shape[1],gt.shape[2])
        avg_feature = avg_feature.mul(mask)

        object_loss = torch.nn.functional.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
        total_object_loss = total_object_loss + object_loss
      
    return total_object_loss
  
class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, _, _, _ = pred.shape
        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)
        class_map = pred.argmax(dim=1).cpu()  # Get Class Map with the Shape: [B, H, W]

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt

        pred_b = F.max_pool2d(
            1 - class_map, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - class_map

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, 2, -1)
        pred_b = pred_b.view(n, 2, -1)
        gt_b_ext = gt_b_ext.view(n, 2, -1)
        pred_b_ext = pred_b_ext.view(n, 2, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
    
def metrics(predictions=None, gts=None, label_values=None, confusion=None):
    label_values = LABELS if label_values is None else label_values
    cm = np.asarray(confusion, dtype=np.int64) if confusion is not None else confusion_matrix(
        gts, predictions, labels=range(len(label_values)))
    if cm.sum() == 0:
        raise ValueError('No valid evaluation pixels')

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:MEAN_CLASS_COUNT])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:MEAN_CLASS_COUNT])
    print('mean MIoU : %.4f' % (MIoU))
    print("---")

    return MIoU
