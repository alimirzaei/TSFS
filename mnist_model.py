from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import keras
import keras.backend as K

def getCodes(X):
    num_classes = 10
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid', name='hidden'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1,
    #           validation_data=(x_test, y_test))
    # model.save_weights('mnist.hd5')
    model.load_weights('mnist_2d.hd5')
   
    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.get_layer('hidden').output)
    codes = intermediate_layer_model.predict(X)


    return codes