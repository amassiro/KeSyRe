#
# KeSyre class
#
#
#


from keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU, Add, Concatenate, Dot
from keras.models import Model
from keras.utils  import plot_model

from keras import backend as K

from keras.layers import Layer

from keras.optimizers  import Adam


class classKeSyRe:


    def __init__(self):
      
      print ("constructor")
      
      self._additive_functions = {}
      self._nested_functions = {}
      self._inputs_dimension = None
      self._multiplicative_branches = None
      self._model = None
      
      
      
    def Print(self):
    
      print ("-------------------")
      print ("KeSyRe is made of: ")
      if self._inputs_dimension is not None :
        print (" inputs dimension = ", self._inputs_dimension)
      if self._additive_functions is not None :
        print (" additive functions = ", self._additive_functions)
      if self._nested_functions is not None :
        print (" nested functions = ", self._nested_functions)
      if self._multiplicative_branches is not None :
        print (" multiplicative branches = ", self._multiplicative_branches)
        
      if self._model is not None :
        self._model.summary()

        plot_model(
            self._model,
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


      
      print ("-------------------")


    def PrintFormula(self):
    
      print ("Best formula = ")
      
      #    for key in self._additive_functions:
      #      #print ("+".join(additive_function for additive_function in self._additive_functions[key]))
      #      for additive_function in self._additive_functions[key] :
      #        for idim in range(self._inputs_dimension) :
      #          print ("A * " + additive_function + "(x_" + str(idim) + ",)")
      
      for layer in self._model.layers: 
        print ("    " + layer.name + " --> " + str( layer.get_weights() ) )
        #print ("                 " + str(layer.get_config()))
        
     
      
    def setInputs(self, inputs):
    
      self._inputs = inputs


    def setInputsDimension(self, dimension):
    
      self._inputs_dimension = dimension



    def defineAdditiveFunctions(self, functions, name):
    
      # boh ... to be thought ...
      print (" additive functions = ", functions)
      
      self._additive_functions[name] = functions


    def defineMultiply(self, names):
    
      # boh ... to be thought ...
      print (" multiplicative branches = ", names)
      
      self._multiplicative_branches = names



    def defineNestedFunctions(self, functions, name):
    
      # boh ... to be thought ...
      print (" nested functions = ", functions)
      
      self._nested_functions[name] = functions



    def compileModel(self):

      #optimizzatore = Adam( learning_rate = 1 )
      optimizzatore = Adam( lr = 0.05 )
      
      self._model.compile(
             loss='MSE', 
             optimizer = optimizzatore
             )
      
      
      
    def createModel(self):
      
      inputs = Input(shape=(self._inputs_dimension,))
      
      hiddens_additive = {}
      concatenated_layer = {}
      layer_of_sum = {}
      hiddens_nested = {}
      
      
      ##---
      for key in self._additive_functions:
        
        hiddens_additive[key] = []
        for additive_function in self._additive_functions[key]:
          if additive_function == "sin" :
            hiddens_additive[key].append (
              MyFunctionSin(
                         beta = 1.0, 
                         alpha = 0.0, 
                         trainable = True,
                         )(inputs)
              )
          if additive_function == "exp" :
            hiddens_additive[key].append (
              MyFunctionExp(
                         alpha = -0.1, 
                         trainable = True,
                         )(inputs)
              )
            
            
        if (len(hiddens_additive[key]) > 1):
          concatenated_layer[key] = Concatenate(axis=1)(hiddens_additive[key])
        else :
          concatenated_layer[key] = hiddens_additive[key][0]
          
        layer_of_sum[key] = Dense(
                1,
                activation='linear',
                use_bias=False
               ) (concatenated_layer[key])

        
        if key in self._nested_functions.keys():
          hiddens_nested[key] = []
          for nested_function in self._nested_functions[key]:
            if nested_function == "exp" :
              hiddens_nested[key].append (
                MyFunctionExp(
                   alpha = -0.1, 
                   trainable = True,
                   )(layer_of_sum[key])
                )
            if nested_function == "sin" :
              hiddens_nested[key].append (
                MyFunctionSin(
                   beta = 1.0, 
                   alpha = 0.0, 
                   trainable = True,
                   )(layer_of_sum[key])
                )
          
          if (len(hiddens_nested[key]) > 1):
            concatenated_layer_temp = Concatenate(axis=1)(hiddens_nested[key])
          else :
            concatenated_layer_temp = hiddens_nested[key][0]
          
          # recycle the layer_of_sum[key] variable
          layer_of_sum[key] = concatenated_layer_temp

        # temporary setting of output
        # useful if only 1 "branch"
        outputs = layer_of_sum[key] 

      ##---
 
      branches_to_be_multiplied = []
              
      for branch in self._multiplicative_branches:
        
        branches_to_be_multiplied.append( layer_of_sum[branch] )
        
      if ( len(branches_to_be_multiplied) > 1 ) :
        layer_of_multiplication = Dot(axes=1)(branches_to_be_multiplied)

        outputs = Dense(
                        1,
                        activation='linear',
                        use_bias=False
                       ) (layer_of_multiplication)
      
      
      # now the output ...  if not already set    
      #else :      
        #outputs = Dense(
                        #1,
                        #activation='linear',
                        #use_bias=False
                       #) (layer_of_multiplication)
        
      # and finally the model 
      
      self._model = Model(inputs=inputs, outputs=outputs)
      





      
      


#------------------------------------------------------------------------------------


def myFunctionSin(x, beta=1.0, alpha=0.0):
   return K.sin(beta * x - alpha)

 

#
# https://keras.io/api/layers/base_layer/
#
   
   
class MyFunctionSin(Layer):

    def __init__(self, beta=1.0, alpha=0.0, trainable=False, **kwargs):
        super(MyFunctionSin, self).__init__(**kwargs)
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

        super(MyFunctionSin, self).build(input_shape)


    def call(self, inputs, mask=None):
        return myFunctionSin(inputs, self.beta_factor, self.alpha_factor)


    def get_config(self):
        config = {
                  'beta' : self.get_weights()[0] if self.trainable else self.beta,
                  'alpha': self.get_weights()[1] if self.trainable else self.alpha,
                  'trainable': self.trainable
                  }
        base_config = super(MyFunctionSin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def compute_output_shape(self, input_shape):
        return input_shape



#------------------------------------------------------------------------------------

def myFunctionExp(x, alpha=0.0):
    return K.exp(alpha * x)

class MyFunctionExp(Layer):

    def __init__(self, alpha=0.0, trainable=False, **kwargs):
        super(MyFunctionExp, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha
        self.trainable = trainable
        self.__name__ = 'MyFunzionissimaExp'


    def build(self, input_shape):
        self.alpha_factor = K.variable(self.alpha,
                                      dtype=K.floatx(),
                                      name='alpha_factor')

        if self.trainable:
            self._trainable_weights.append(self.alpha_factor)

        super(MyFunctionExp, self).build(input_shape)


    def call(self, inputs, mask=None):
        return myFunctionExp(inputs, self.alpha_factor)


    def get_config(self):
        config = {
                  'alpha': self.get_weights()[0] if self.trainable else self.alpha,
                  'trainable': self.trainable
                  }
        base_config = super(MyFunctionExp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def compute_output_shape(self, input_shape):
        return input_shape
   
        
         
   
         