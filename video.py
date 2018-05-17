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

#helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

#def process_image(image, sess, logits, keep_prob, image_pl, image_shape):
def process_image(image):
    image = scipy.misc.imresize(image, image_shape)

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)


#img = process_image(scipy.misc.imread('./data/data_road/testing/image_2/um_000000.png'), sess, logits, keep_prob, image_pl, image_shape)
#scipy.misc.imsave('test_img.png', img)

video_output = 'result.mp4'
clip1 = VideoFileClip('project_video.mp4')
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(video_output, audio=False)
