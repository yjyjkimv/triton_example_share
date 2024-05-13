import numpy as np
import cv2
import argparse

from tritonclient.utils import *
import tritonclient.grpc as grpcclient

def parser_initialize(parser):

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="core",
        help="Model name",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="0.0.0.0:8001",
        help="Inference server URL. Default is 0.0.0.0:8001.",
    )

    parser.add_argument(
        "--image",
        type=str,
        required=False,
        default="sample_0004_00021.png",
        help="Path to the image",
    )

    return parser

def image_read(IMAGE_PATH):

    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def preprocess_example(image):

    input_image = cv2.resize(image, (640, 640))
    input_image = input_image.astype('float32')
    input_image = input_image.transpose((2,0,1))[np.newaxis, :] / 255.0
    input_image = np.ascontiguousarray(input_image)
    
    return input_image


def postprocess_example(detected):

    # do something with
    # detected

    result = 1.0

    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = parser_initialize(parser)
    args = parser.parse_args()

    ## print arguments 
    print('args.model_name: ', args.model_name)
    print('args.url: ', args.url)
    print('args.image: ', args.image)

    MODEL_NAME = args.model_name
    SERVER_URL = args.url
    IMAGE_PATH = args.image

    # read image 
    image = image_read(IMAGE_PATH)
    print("## input image shape : ", image.shape)

    # preprocess 
    input_image = preprocess_example(image)
    print("## preprocessed image shape : ", input_image.shape)

    with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
        inputs = [
            grpcclient.InferInput("images", input_image.shape, np_to_triton_dtype(np.float32))
        ]
        inputs[0].set_data_from_numpy(input_image)
        outputs = [
            grpcclient.InferRequestedOutput("output0")
        ]
        response = triton_client.infer(
                                    model_name=MODEL_NAME,
                                    inputs=inputs,
                                    outputs=outputs
                                    )
        response.get_response()
        detected = response.as_numpy("output0")

    print('## inferenced result shape: ', detected.shape)

    result = postprocess_example(detected)
    print("## postprocessed result : ", result)

    