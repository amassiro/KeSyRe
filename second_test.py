import matplotlib.pyplot as plt
import numpy as np


def f(t):
    #return np.exp(-t) * np.sin(2*np.pi*t)
    #return 3.5 * np.sin(2*t)
    return 3.5 * np.sin(2*t) + 0.2 * np.sin(20*t)


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

from keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU, Add, Concatenate
from keras.models import Model
from keras.utils  import plot_model

#
# https://datascience.stackexchange.com/questions/58884/how-to-create-custom-activation-functions-in-keras-tensorflow
#

from keras import backend as K

def myFunction(x, beta=1.0, alpha=0.0):
    return K.sin(beta * x - alpha)
 
 
#
# https://keras.io/api/layers/base_layer/
#

from keras.layers import Layer

class MyFunction(Layer):

    def __init__(self, beta=1.0, alpha=0.0, trainable=False, **kwargs):
        super(MyFunction, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.alpha = alpha
        self.trainable = trainable
        self.__name__ = 'MyFunzionissima'


    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta,
                                      dtype=K.floatx(),
                                      name='beta_factor')
        self.alpha_factor = K.variable(self.alpha,
                                      dtype=K.floatx(),
                                      name='alpha_factor')

        if self.trainable:
            self._trainable_weights.append(self.beta_factor)
            self._trainable_weights.append(self.alpha_factor)

        super(MyFunction, self).build(input_shape)


    def call(self, inputs, mask=None):
        return myFunction(inputs, self.beta_factor, self.alpha_factor)


    def get_config(self):
        config = {
                  'beta' : self.get_weights()[0] if self.trainable else self.beta,
                  'alpha': self.get_weights()[1] if self.trainable else self.alpha,
                  'trainable': self.trainable
                  }
        base_config = super(MyFunction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def compute_output_shape(self, input_shape):
        return input_shape




#import tensorflow as tf
#from keras.utils.generic_utils import get_custom_objects
#get_custom_objects().update({'myFunction': tf.keras.layers.Activation(myFunction)})




# You then would add the activation function the same as any other layer:

#                   1 ---> inputs has only 1 dimension
inputs=Input(shape=(1,))

#
# sum of many "sin"
#

#hidden = Dense(
               #1,
               #activation = MyFunction(beta=2.0, alpha=0.0, trainable=True),
               #use_bias=False
               #)(inputs)

hidden1 = MyFunction(
               beta = 2.0, 
               alpha = 0.0, 
               trainable = True,
               )(inputs)


#from keras.constraints  import MinMaxNorm

hidden2 = MyFunction(
               beta = 18.1, 
               alpha = 0.0, 
               trainable = True,
               #kernel_constraint = MinMaxNorm(min_value=10.0, max_value=30.0)
               )(inputs)


concatenated_layer = Concatenate(axis=1)([hidden1, hidden2])

#model.add(MyFunction(beta=1.0, trainable=True))




#
# then add all together with weights ...
#

outputs = Dense(
                1,
                activation='linear',
                use_bias=False
               ) (concatenated_layer)


model = Model(inputs=inputs, outputs=outputs)


#     tf.keras.optimizers.Adam(
#         learning_rate=0.001,
#         beta_1=0.9,
#         beta_2=0.999,
#         epsilon=1e-07,
#         amsgrad=False,
#         name="Adam",
#         **kwargs
#     )


from keras.optimizers  import Adam

#optimizzatore = Adam( learning_rate = 1 )
optimizzatore = Adam( lr = 0.5 )


model.compile(
       loss='MSE', 
       #optimizer='adam'
       optimizer = optimizzatore
       )

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

x_axis = np.arange(0.0, 5.0, 0.01)

X_train = x_axis
Y_train = f(x_axis)

X_validation = x_axis
Y_validation = f(x_axis)

#print ("X_train = ", X_train)
#print ("Y_train = ", Y_train)


history = model.fit(
                     X_train,
                     Y_train,
                     validation_data = (X_validation,Y_validation),
                     epochs=1000,
                     verbose=0
                    )



print ( history.history.keys() )


for layer in model.layers: print(layer.get_config(), layer.get_weights())
ilayer=0
for layer in model.layers: 
  print (" layer ", ilayer)
  ilayer +=1
  print(" get_weights ---> " , layer.get_weights())

#model.layers[0].weights


#plt.plot(history.history["val_loss"])
#plt.plot(history.history["loss"])
#plt.show()


Y_predicted_validation = model.predict(X_validation)



plt.plot(X_validation, Y_validation, "b .")
plt.plot(X_validation, Y_predicted_validation, "r +")

plt.show()




