import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder as OHE


class LITE:
    def __init__(
        self,
        output_directory,
        length_TS,
        n_classes,
        batch_size=64,
        n_filters=32,
        kernel_size=41,
        n_epochs=1500,
        verbose=True,
        use_custom_filters=True,
        use_dilation=True,
        use_multiplexing=True,
    ):

        self.output_directory = output_directory

        self.length_TS = length_TS
        self.n_classes = n_classes

        self.verbose = verbose

        self.n_filters = n_filters

        self.use_custom_filters = use_custom_filters
        self.use_dilation = use_dilation
        self.use_multiplexing = use_multiplexing

        self.kernel_size = kernel_size - 1

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.compile()


    def hybird_layer(self, input_tensor, input_channels, kernel_sizes=np.array([[2, 4, 8, 16, 32, 64]])):
        conv_list = []  

        for kernel_size in kernel_sizes:
            filter_ = np.ones(shape=(kernel_size, input_channels, 1))
            filter_[np.arange(kernel_size) % 2 == 0] *= -1
            
            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False
            )(input_tensor)
            conv_list.append(conv) 

        for kernel_size in kernel_sizes:
            filter_ = np.ones(shape=(kernel_size, input_channels, 1))
            filter_[np.arange(kernel_size) % 2 != 0] *= -1
            
            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False
            )(input_tensor)
            conv_list.append(conv)  

        for kernel_size in kernel_sizes[1:]:
            filter_ = np.zeros(shape=(kernel_size + kernel_size // 2, input_channels, 1))
            xmash = np.linspace(start=0, stop=1, num=kernel_size // 4 + 1)[1:].reshape((-1, 1, 1))
            filter_[:kernel_size // 4] = -xmash**2
            filter_[kernel_size // 4:kernel_size // 2] = -np.flip(filter_[:kernel_size // 4])
            filter_[kernel_size // 2:3 * kernel_size // 4] = 2 * xmash**2
            filter_[3 * kernel_size // 4:kernel_size] = 2 * np.flip(filter_[kernel_size // 2:3 * kernel_size // 4])
            filter_[kernel_size:5 * kernel_size // 4] = -xmash**2
            filter_[5 * kernel_size // 4:] = -np.flip(filter_[:kernel_size // 4])
            
            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size + kernel_size // 2,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False
            )(input_tensor)
            conv_list.append(conv)  

        hybird_layer = tf.keras.layers.Concatenate(axis=2)(conv_list)
        hybird_layer = tf.keras.layers.Activation(activation="relu")(hybird_layer)

        return hybird_layer

    def compile(self):
        input_shape = (self.length_TS,)
        input_layer = tf.keras.layers.Input(input_shape)

        n_convs = 1
        n_filters = self.n_filters * 3
        kernel_size_s = [self.kernel_size // (2**i) for i in range(n_convs)]

        input_layer = tf.keras.layers.Reshape(target_shape=(self.length_TS, 1))(
            input_layer
        )

        conv_list = []
        for i in range(len(kernel_size_s)):
            conv_list.append(
                tf.keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=1,
                    padding="same",
                    dilation_rate=1,
                    activation="relu",
                    use_bias=False,
                )(input_layer)
            )
        self.hybird = self.hybird_layer(
                input_tensor=input_layer, input_channels=input_layer.shape[-1]
            )
        conv_list.append(self.hybird)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation="relu")(x)
        self.kernel_size //= 2

        input_tensor = x

        dilation_rate = 1
        for i in range(2):

            if self.use_dilation:
                dilation_rate = 2 ** (i + 1)

            x = self._fcn_module(
                input_tensor=input_tensor,
                kernel_size=self.kernel_size // (2**i),
                n_filters=self.n_filters,
                dilation_rate=dilation_rate,
            )

            input_tensor = x

        gap = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(
            units=self.n_classes, activation="softmax"
        )(gap)

        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=50, min_lr=1e-4
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.output_directory + "best_model.hdf5",
            monitor="loss",
            save_best_only=True,
        )
        self.callbacks = [reduce_lr, model_checkpoint]

        self.model.compile(
            loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )

        return self.model
        

    def fit(self, xtrain, ytrain, xval, yval):
        ohe = OHE(sparse=False)

        ytrain = np.expand_dims(ytrain, axis=1)
        ytrain = ohe.fit_transform(ytrain)

        yval = np.expand_dims(yval, axis=1)
        yval = ohe.fit_transform(yval)

        hist = self.model.fit(
                xtrain,
                ytrain,
                batch_size=self.batch_size,
                epochs=self.n_epochs,
                verbose=self.verbose,
                validation_data=(xval, yval),
                callbacks=self.callbacks,
            )
        return hist

