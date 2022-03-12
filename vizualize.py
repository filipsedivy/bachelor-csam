from toolkits.preprocessing import load_img, img_to_array
from toolkits.gradcam import grad_cam_plus
from keras.models import load_model
from PIL import Image
import numpy as np
import sys
import cv2


def show_image(src, heatmap) -> None:
    image = cv2.imread(src)

    image_heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    image_heatmap = (image_heatmap * 255).astype('uint8')
    image_heatmap = cv2.applyColorMap(image_heatmap, cv2.COLORMAP_JET)

    # 0.5 is image alpha channel
    combine = image_heatmap * 0.5 + image
    combine = np.clip(combine, 0, 255).astype('uint8')
    combine = cv2.cvtColor(combine, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(combine)
    img.show()


def run(model: str, image_src: str) -> None:
    model = load_model(model)
    image = load_img(image_src, target_size=(224, 224))
    image = img_to_array(image)

    heatmap = grad_cam_plus(model, image)
    show_image(image_src, heatmap)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Bad parameters: vizualize.py <model.h5> <picture.jpg>')
        exit(255)

    arg_model = sys.argv[1]
    arg_image = sys.argv[2]

    print('Model:', arg_model)
    print('Image:', arg_image)

    run(arg_model, arg_image)
