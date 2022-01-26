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

#
#     x     layer-multiplication     layer-sum
#     x
#     x     layer-division           layer-sum
#     x
#     x     layer-division           layer-multiplication     layer-sum
#     x
#


#
#    Additive --> use a dense with linear activation function
#

#  e.g.
#     A*B + C*D     
#     (A+B)*C   ---> it's associative, it's the same!
#     A/B + C/D  
#     A/B*C + A/B*D + A/C*D + ...
#
  
  
      self._layer_sums_inputs = None
      self._layer_product_inputs = None

      
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



    def compileModel(self, myloss='MSE', optimizzatore = Adam( lr = 0.05 )):

      #optimizzatore = Adam( learning_rate = 1 )
      #optimizzatore = Adam( lr = 0.05 )
      
      self._model.compile(
             loss=myloss, 
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
          if additive_function == "cos" :
            hiddens_additive[key].append (
              MyFunctionCos(
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
          if additive_function == "log" :
            hiddens_additive[key].append (
              MyFunctionLog(
                         alpha = 0.1, 
                         trainable = True,
                         )(inputs)
              )
          if additive_function == "pow" :
            hiddens_additive[key].append (
              MyFunctionPow(
                         alpha = 0.1, 
                         trainable = True,
                         )(inputs)
              )
          if additive_function == "sqrt" :
            hiddens_additive[key].append (
              MyFunctionSqrt(
                         alpha = 0.1, 
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
            if nested_function == "log" :
              hiddens_nested[key].append (
                MyFunctionLog(
                   alpha = 0.1, 
                   trainable = True,
                   )(layer_of_sum[key])
                )
            if nested_function == "pow" :
              hiddens_nested[key].append (
                MyFunctionPow(
                   alpha = -0.1, 
                   trainable = True,
                   )(layer_of_sum[key])
                )
            if nested_function == "sqrt" :
              hiddens_nested[key].append (
                MyFunctionSqrt(
                   alpha = 0.1, 
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
            if nested_function == "cos" :
              hiddens_nested[key].append (
                MyFunctionCos(
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
      




#
# https://github.com/keras-team/keras/issues/890
# 

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)
  
  
  
