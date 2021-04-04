import tensorflow as tf
import tensorflow_addons as tfa

def random_translate(imgs, max_translate=6):
    n = imgs.shape[0]

    translations = tf.random.uniform((n, 2), 
        -max_translate, max_translate)
    imgs = tfa.image.translate(imgs, translations)
    
    return imgs

