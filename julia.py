import numpy as np
import tensorflow as tf
from PIL import Image
from moviepy.editor import ImageSequenceClip

_R = 4
_ITER_NUM = 200

# Julia Set {c = constant; z1 = z; z2 = z1^2+c; zn+1 = zn^2 + c}
def julia(Z, c, bratio, ratio):

    def color(bgratio, ratio):    
        def _color(z, step):
            if abs(z) < _R:
                return 0, 0, 0
            v = np.log2(step + _R - np.log2(np.log2(abs(z)))) / 5
            if v < 1.0:
                return v**bgratio[0], v**bgratio[1], v**bgratio[2]
            else:
                v = max(0, 2 - v)
            return v**ratio[0], v**ratio[1], v**ratio[2]
        return _color

    xs = tf.constant(np.full(shape=Z.shape, fill_value=c, dtype=Z.dtype)) # c = input c
    zs = tf.Variable(Z)                                                   # init z1 = input Z
    ns = tf.Variable(tf.zeros_like(xs, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        zs_ = tf.where(tf.abs(zs) < _R, zs ** 2 + xs, zs)
        is_diverged = tf.abs(zs_) < _R
        step = tf.group(
            zs.assign(zs_),
            ns.assign_add(tf.cast(is_diverged, tf.float32))
        )
        for _ in range(_ITER_NUM):
            step.run()
        final_step = ns.eval()
        final_z = zs_.eval()
    r, g, b = np.frompyfunc(color(bratio, ratio), 2, 3)(final_z, final_step)
    img = np.dstack((r, g, b))
    return Image.fromarray(np.uint8(img * 255))

def default(is_save=False, ipath='img/'):
    bgratio = (4, 2.5, 1)   # background rgb ratio
    ratio = (0.9, 0.9, 0.9) # fractal graph rgb ratio
    lft, rgt = -2.5, 1.5    # x range
    top, btm = -1.2, 1.2    # y range
    width = 1000            # image width
    step = (rgt - lft) / width
    Y, X = np.mgrid[top:btm:step, lft:rgt:step]
    Z = X + 1j * Y
    c = 0.285 + 0.01j
    img = julia(Z, c, bgratio, ratio)
    if is_save:
        img.save(ipath+'julia.png')
    else:
        img.show()

def gif(fname, array, fps=10, scale=1.0):
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(fname, fps=fps)
    return clip

def dynamic_julia(n=60, ipath='img/'):
    bgratio = (4, 2.5, 1)   # background rgb ratio
    ratio = (0.9, 0.9, 0.9) # fractal graph rgb ratio
    lft, rgt = -2.5, 1.5    # x range
    top, btm = -1.2, 1.2    # y range
    width = 1000            # image width
    step = (rgt - lft) / width
    Y, X = np.mgrid[top:btm:step, lft:rgt:step]
    Z = X + 1j * Y

    seqs = np.zeros([n] + list(Z.shape) + [3])
    for i in range(n):
        theta = 2 * np.pi / n * i
        c = -(0.835 - 0.1 * np.cos(theta)) - (0.2321 + 0.1 * np.sin(theta)) * 1j
        img = julia(Z, c, bgratio, ratio)
        seqs[i, :, :] = np.array(img)
        print('generate img {}'.format(i))
    gif(ipath+'julia.gif', seqs)

if __name__ == '__main__':
    # default()
    dynamic_julia()
