# coding=utf-8
import cv2
import math
import os
import tarfile

import numpy as np
import tensorflow as tf

"""
filepath label xmin ymin xmax ymax
./20167/3776065B_16_D_215.jpg 1  60  42  188  228

"""

DIRECTORY = os.path.join(os.path.dirname(__file__), 'annotation')

flags = tf.app.flags
flags.DEFINE_string('output_dir', os.path.dirname(__file__), 'Path to directory to output TFRecords.')
FLAGS = flags.FLAGS


def write_record2(writer, img, filepath, data):
    """
    画像一枚ずつをTFRecord形式にしている?
    :param writer:
    :param img:
    :param filepath:
    :param data:
    :return:
    """
    h, w, _ = img.shape
    xmin, xmax, ymin, ymax = [], [], [], []
    class_text, class_label = [], []
    label_map_dict = {
        'tops': 1,
        'bottoms': 2
    }

    label = label_map_dict[data['class']]
    xmin.append(data['xmin'] / w)
    xmax.append(data['xmax'] / w)
    ymin.append(data['ymin'] / h)
    ymax.append(data['ymax'] / h)
    class_text.append(data['class'].encode('utf-8'))
    class_label.append(label)
    with open(filepath, 'rb') as f:
        encoded = f.read()
    print(xmin)
    print(data['xmin'])
    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filepath.encode('utf-8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filepath.encode('utf-8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf-8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=class_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=class_label)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def main(argv=None):
    writers = {
        'train': tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, 'clothes_train.record')),
        'val': tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, 'clothes_val.record')),
    }
    folds_dir = DIRECTORY
    for filename in os.listdir(folds_dir):
        if not 'annot' in filename:
            continue
        writer = writers['val'] if 'val' in filename else writers['train']
        with open(os.path.join(folds_dir, filename)) as f:
            for line in f:
                data = line.rstrip().split()
                print(data)
                img_file = data[0]
                if data[1] == 0 or data[1] == '0':
                    data[1] = 'tops'
                else:
                    data[1] = 'bottoms'
                result = {
                    'class': data[1],
                    'xmin': float(data[2]),
                    'ymin': float(data[3]),
                    'xmax': float(data[4]),
                    'ymax': float(data[5])
                }
                # try:
                if len(img_file) == 0:
                    break
                # load image, detect faces
                img = cv2.imread(img_file)
                write_record2(writer, img, img_file, result)

                # for results in detected:
                #     for obj in results:
                #         cv2.rectangle(
                #             img,
                #             tuple([int(obj['xmin'] + .5), int(obj['ymin'] + .5)]),
                #             tuple([int(obj['xmax'] + .5), int(obj['ymax'] + .5)]),
                #             [0, 255, 0] if obj['class'] == 'face' else [255, 255, 0]
                #         )
                # cv2.imshow(img_file, img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # except:
                #     print('error occurred.')
                #     pass
    for writer in writers.values():
        writer.close()


if __name__ == '__main__':
    # import os
    # print(os.path.dirname(__file__))
    # print(os.path.join(os.path.dirname(__file__), 'clothes'))
    # print(os.listdir(os.path.join(os.path.dirname(__file__), 'clothes')))
    tf.app.run()
