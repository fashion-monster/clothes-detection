from flask import Flask, request
from fashion_detection_tutorial import fashion_detector, image_load
import requests

app = Flask(__name__)


@app.route("/cloth_detect", methods=['POST'])
def cloth_detection_facet():
    """clothe area detect method.

    Returns:

    """
    image_name = request.form["image_path"]
    path = '/home/hashimoto/LineBot' + image_name
    image = image_load(path=path)
    file_name = path.split('/')
    cropped_dir_name = '/home/hashimoto/LineBot/tmp/cropped'
    output_name = cropped_dir_name + file_name[-1]

    rslt = fashion_detector(pil_image=image, output_name=output_name)
    if rslt is not None:
        header = {'content-type': 'application/json'}
        print(requests.post(url='http://127.0.0.1:9999/resize',
                            headers=header,
                            data="{'image_path':'" + output_name + "'}"))
    else:
        print('failed detect')
    return


if __name__ == '__main__':
    app.run(port=9998, debug=True)
