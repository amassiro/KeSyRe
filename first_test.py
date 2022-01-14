import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


print ("test with a simple function")

t1 = np.arange(0.0, 5.0, 0.01)

#plt.plot(t1, f(t1), "b .")

#plt.show()


#
# this is the target function
#
#


print ("now keras")

#
# now keras
# 

from keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU, Add
from keras.models import Model
from keras.utils  import plot_model

from keras import backend as K
def my_custom_activation(x):
    return (K.sigmoid(x) * 5) - 1



#                   1 ---> inputs has only 1 dimension
inputs=Input(shape=(1,))

hidden=Dense(
  10,
  activation = my_custom_activation
  )(inputs)
#hidden=Dense(1500, activation='sigmoid')(inputs)
#hidden=Dense(1500, activation='relu')(inputs)

#outputs = tf.keras.layers.Add()([hidden])
#outputs = Add()([hidden])
outputs = Dense(
                1,
                activation='linear',
                use_bias=False
               ) (hidden)


 
#outputs = Dense(1)(hidden)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='MSE', optimizer='adam')

model.summary()


plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    #show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    #expand_nested=True,
    #dpi=96,
    #layer_range=None,
    #show_layer_activations=True,
)



#
# train
# 

x_axis = np.arange(0.0, 5.0, 0.1)

X_train = x_axis
Y_train = f(x_axis)

X_validation = x_axis
Y_validation = f(x_axis)

print ("X_train = ", X_train)
print ("Y_train = ", Y_train)

##print(X_tr[:10])
##print(y_tr[:10])
##print(X_val[:10])
##print(y_val[:10])

history = model.fit(
                     X_train,
                     Y_train,
                     validation_data = (X_validation,Y_validation),
                     epochs=150,
                     verbose=0
                    )

print ( history.history.keys() )

#plt.plot(history.history["val_loss"])
#plt.plot(history.history["loss"])
#plt.show()


Y_predicted_validation = model.predict(X_validation)



plt.plot(X_validation, Y_validation, "b .")
plt.plot(X_validation, Y_predicted_validation, "r +")

plt.show()




