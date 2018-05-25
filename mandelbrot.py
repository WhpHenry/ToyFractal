import numpy as np
import tensorflow as tf
from PIL import Image

_R = 4
_ITER_NUM = 200
# reference: https://github.com/hzy46/tensorflow-fractal-playground
# Mandelbrot Set {z0 = 0; z1 = c; zn+1 = zn^2 + c}
def mandelbrot(Z, bgratio, ratio):

    def color(bgratio, ratio):    
        def _color(z, step):
            if abs(z) < _R:
                return 0, 0, 0
            v = np.log2(step + _R - np.log2(np.log2(abs(z)))) / 5
            if v < 1.0:    # background
                return v**bgratio[0], v**bgratio[1], v**bgratio[2]
            else:
                v = max(0, 2 - v)
            return v**ratio[0], v**ratio[1], v**ratio[2]
        return _color

    xs = tf.constant(Z.astype(np.complex64))  # c = input Z
    zs = tf.Variable(xs)                      # init z1 = c
    ns = tf.Variable(tf.zeros_like(xs, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tf.where(m,x,y) for matrix m if m(i,j) = x(i,j) else y(i,j)
        zs_ = tf.where(tf.abs(zs) < _R, zs ** 2 + xs, zs) 
        is_diverged = tf.abs(zs_) < _R # bool matrix
        step = tf.group(
            zs.assign(zs_),
            ns.assign_add(tf.cast(is_diverged, tf.float32))
        )
        for _ in range(_ITER_NUM):
            step.run()
        final_step = ns.eval()  # bool matrix
        final_z = zs_.eval()    # value pixal matrix
    # print(final_step)
    r, g, b = np.frompyfunc(color(bgratio, ratio), 2, 3)(final_z, final_step)
    img = np.dstack((r, g, b))
    return Image.fromarray(np.uint8(img * 255))  

def default(is_save=False, ipath='img/'):
    bgratio=(4,2.5,1)     # background rgb ratio
    ratio=(1,1.5,3)       # fractal graph rgb ratio
    lft, rgt = -2.5, 1.5  # x range
    top, btm = -1.2, 1.2  # y range
    width = 1000          # image width
    step = (rgt - lft) / width
    Y, X = np.mgrid[top:btm:step, lft:rgt:step]
    Z = X + 1j * Y
    img = mandelbrot(Z, bgratio, ratio)
    if is_save:
        img.save(ipath+'mandelbrot.png')
    else:
        img.show()

def area(is_save=False, ipath='img/'):
    # # Elephant Valley
    # bgratio=(4,2.5,1)       # background rgb ratio
    # ratio=(0.9, 0.6, 0.6)   # fractal graph rgb ratio
    # lft, rgt = 0.275, 0.28  # x range   
    # top, btm = 0.006, 0.01  # y range

    # # Seahorse Valley
    # bgratio=(4,2.5,1)       # background rgb ratio
    # ratio=(0.1, 0.1, 0.3)   # fractal graph rgb ratio
    # lft, rgt = -0.750, -0.747  # x range   
    # top, btm = 0.099, 0.102  # y range

    # Triple Spiral Valley
    bgratio=(4,2.5,1)           # background rgb ratio
    ratio=(0.2, 0.6, 0.6)       # fractal graph rgb ratio
    lft, rgt = -0.090, -0.086   # x range   
    top, btm = 0.654, 0.657     # y range

    width = 1000          # image width
    step = (rgt - lft) / width
    Y, X = np.mgrid[top:btm:step, lft:rgt:step]
    Z = X + 1j * Y
    img = mandelbrot(Z, bgratio, ratio)
    if is_save:
        img.save(ipath+'mandelbrot_area.png')
    else:
        img.show()

if __name__ == '__main__':
    # default()
    area()
