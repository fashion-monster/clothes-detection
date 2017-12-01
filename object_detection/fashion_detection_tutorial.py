# coding: utf-8


def image_load(path):
    """ this return PIL Image from path.


    cv2 version is following
    Args:
        path:

    Returns:
        image
    """
    from PIL import Image
    image = Image.open(path)

    #
    # import cv2
    # image = cv2.imread(path)
    return image


def fashion_detect(image):
    """ this detects clothes area and label. Maybe it' return only pants

    Args:
        image:

    Returns:
        box:
        label:
    """
    # # Imports
    import numpy as np
    import os
    import sys
    import tensorflow as tf

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    # ## Object detection imports
    # Here are the imports from the object detection module.
    from utils import label_map_util
    from utils import visualization_utils as vis_util

    # # Model preparation
    MODEL_NAME = 'cloth_export'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'clothes_label_map.pbtxt')

    NUM_CLASSES = 2

    # ## Load a (frozen) Tensorflow model into memory.

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ## Loading label map Label maps map indices to category names, so that when our convolution network predicts
    # `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that
    # returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # ## Helper code
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    # # Detection
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            left, right, top, bottom, label = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            box = {'left': int(left), 'right': int(right), 'top': int(top), 'bottom': int(bottom)}
            return box, label


def image_crop(image, box, name):
    """ cropping from image with box.

    Args:
        image:
        box:

    Returns:

    """
    croped_image = image.crop((box['left'], box['top'], box['right'], box['bottom']))
    croped_image.save(name)
    return True


def fashion_detector(pil_image, output_name):
    """ facets method. crop image and save it with it's name.

    Args:
        pil_image:PIL image

    Returns:
        label
    """
    box, label = fashion_detect(pil_image)
    image_crop(pil_image, box, output_name)
    return str(label[0])


if __name__ == '__main__':
    image = image_load('test_images/image4.jpg')
    print(fashion_detector(pil_image=image, output_name="croped5.jpg"))
