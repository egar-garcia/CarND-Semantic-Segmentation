import sys

# Checking the number of command line arguments
if len(sys.argv) < 3:
    print("Syntax: python3 video.py input_video_file output_video_file");
    sys.exit(0)


import tensorflow as tf
import helper
import numpy as np
import scipy.misc
from moviepy.editor import VideoFileClip


num_classes = 2
image_shape = (160, 576)
data_dir = './data'
runs_dir = './runs'
model_dir = './model'
model_meta = './model/model.meta'

sess = tf.Session()

saver = tf.train.import_meta_graph(model_meta)
saver.restore(sess, tf.train.latest_checkpoint(model_dir))

graph = tf.get_default_graph()
correct_label = graph.get_tensor_by_name('correct_label:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
image_pl = graph.get_tensor_by_name('image_input:0')
logits = graph.get_tensor_by_name('logits:0')


def process_image(image):
    """
    Uses the trained FCN to identify the road surface
    and return an image with that highlighted
    """
    original_shape = image.shape
    # Scales the image to the input size used by the FCN
    image = scipy.misc.imresize(image, image_shape)

    # Identifying the road surface
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)

    # Highlighting the identified pixels in the image
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    # Resizes the image back to its original size
    street_im = scipy.misc.imresize(street_im, original_shape)

    return np.array(street_im)

video_input = sys.argv[1]
video_output = sys.argv[2]

clip1 = VideoFileClip(video_input)
# Generating the output video
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(video_output, audio=False)
