import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from utility.utils import squarest_grid_size


def grid_placed(images, size=None):
    assert len(images.shape) == 4, f'images should be 4D, but get shape {images.shape}'
    B, H, W, C = images.shape
    if size is None:
        size = squarest_grid_size(B)
    image_type = images.dtype
    if (images.shape[3] in (3,4)):
        img = np.zeros((H * size[0], W * size[1], C), dtype=image_type)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * H:j * H + H, i * W:i * W + W, :] = image
        if np.issubdtype(image_type, np.uint8):
            return img
        if np.min(img) < -.5:
            # for images in range [-1, 1], make it in range [0, 1]
            img = (img + 1) / 2
        elif np.min(img) < 0:
            # for images in range [-.5, .5]
            img = img + .5
        assert np.min(img) >= 0, np.min(img)
        assert np.max(img) <= 1, np.max(img)
        img = np.clip(255 * img, 0, 255).astype(np.uint8)
        return img
    elif images.shape[3]==1:
        img = np.zeros((H * size[0], W * size[1]), dtype=image_type)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * H:j * H + H, i * W:i * W + W] = image[:,:,0]
        return img
    else:
        NotImplementedError


def encode_gif(frames, fps):
    """ encode gif from frames in another process, return a gif """
    from subprocess import Popen, PIPE
    H, W, C = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[C]
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {W}x{H} -pix_fmt {pxfmt} -i - -filter_complex',
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out


def save_video(name, video, fps=30):
    name = name if isinstance(name, str) else name.decode('utf-8')
    video = np.array(video, copy=False)
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    while len(video.shape) < 5:
        video = np.expand_dims(video, 0)
    B, T, H, W, C = video.shape
    if B != 1:
        # merge multiple videos into a single video
        bh, bw = squarest_grid_size(B)
        frames = video.reshape((bh, bw, T, H, W, C))
        frames = frames.transpose((2, 0, 3, 1, 4, 5))
        frames = frames.reshape((T, bh*H, bw*W, C))
    else:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, W, C))
    f1, *frames = [Image.fromarray(f) for f in frames]
    if not os.path.isdir('results'):
        os.mkdir('results')
    path = f'results/{name}.gif'
    f1.save(fp=path, format='GIF', append_images=frames,
         save_all=True, duration=1000//fps, loop=0)
    print(f"video is saved to '{path}'")

""" summaries useful for core.log.graph_summary"""
def image_summary(name, images, step=None):
    # when wrapped by tf.numpy_function in @tf.function, str are 
    # represented by bytes so we need to convert it back to str
    name = name if isinstance(name, str) else name.decode('utf-8')
    if len(images.shape) == 3:
        images = images[None]
    if np.issubdtype(images.dtype, np.floating):
        assert np.logical_and(images >= 0, images <= 1).all()
        images = np.clip(255 * images, 0, 255).astype(np.uint8)
    img = np.expand_dims(grid_placed(images), 0)
    tf.summary.image(name + '/image', img, step)

def video_summary(name, video, size=None, fps=30, step=None):
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    while len(video.shape) < 5:
        video = np.expand_dims(video, 0)
    B, T, H, W, C = video.shape
    if size is None and B != 1:
        bh, bw = squarest_grid_size(B)
        frames = video.reshape((bh, bw, T, H, W, C))
        frames = frames.transpose((2, 0, 3, 1, 4, 5))
        frames = frames.reshape((T, bh*H, bw*W, C))
    else:
        if size is None:
            size = (1, 1)
        assert size[0] * size[1] == B, f'Size({size}) does not match the batch dimension({B})'
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, size[0]*H, size[1]*W, C))
    try:
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name, image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + '/image', frames, step)
