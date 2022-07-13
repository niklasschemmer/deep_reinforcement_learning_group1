from tensorflow.keras import layers
import tensorflow as tf

def create_model_actor():

    input_0 = layers.Input(shape=(96, 96, 3))

    conv_0 = layers.Conv2D(8, kernel_size=4, strides=2, activation='relu', kernel_initializer=tf.initializers.glorot_normal(),
                                bias_initializer=tf.initializers.constant(0.1))(input_0)
    conv_1 = layers.Conv2D(16, kernel_size=3, strides=2, activation='relu', kernel_initializer=tf.initializers.glorot_normal(),
                                bias_initializer=tf.initializers.constant(0.1))(conv_0)
    conv_2 = layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', kernel_initializer=tf.initializers.glorot_normal(),
                                bias_initializer=tf.initializers.constant(0.1))(conv_1)
    flat_0 = layers.Flatten()(conv_2)

    dense_0 = layers.Dense(64, activation='relu')(flat_0)
    dense_1 = layers.Dense(6, activation='softplus')(dense_0)
    reshape_0 = layers.Reshape((3, 2))(dense_1)
    lamb_0 = layers.Lambda(lambda x: x + 1)(reshape_0)

    model = tf.keras.Model(inputs=[input_0], outputs=[lamb_0])
    model.compile(optimizer=tf.optimizers.Adam(0.001))

    return model

def create_model_critic():

    input_0 = layers.Input(shape=(96, 96, 3))

    conv_0 = layers.Conv2D(8, kernel_size=4, strides=2, activation='relu', kernel_initializer=tf.initializers.glorot_normal(),
                                bias_initializer=tf.initializers.constant(0.1))(input_0)
    conv_1 = layers.Conv2D(16, kernel_size=3, strides=2, activation='relu', kernel_initializer=tf.initializers.glorot_normal(),
                                bias_initializer=tf.initializers.constant(0.1))(conv_0)
    conv_2 = layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', kernel_initializer=tf.initializers.glorot_normal(),
                                bias_initializer=tf.initializers.constant(0.1))(conv_1)
    flat_0 = layers.Flatten()(conv_2)

    dense_2 = layers.Dense(64, activation='relu')(flat_0)
    dense_3 = layers.Dense(1)(dense_2)

    model = tf.keras.Model(inputs=[input_0], outputs=[dense_3])
    model.compile(optimizer=tf.optimizers.Adam(0.001))

    return model