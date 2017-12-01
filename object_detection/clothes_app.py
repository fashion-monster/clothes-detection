from flask import Flask, request
from fashion_detection_tutorial import fashion_detector, image_load

app = Flask(__name__)


@app.route("/cloth_detect", methods=['POST'])
def cloth_detection_facet():
    """

    Returns:

    """
    path = request.args.get("image_path")
    image = image_load(path=path)
    file_name = path.split('/')
    cropped_dir_name = '/home/hashimoto/LineBot/cropped/'
    return fashion_detector(pil_image=image, output_name=cropped_dir_name + file_name[-1])


if __name__ == '__main__':
    app.run(port=9998)
