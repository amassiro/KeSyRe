
import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("Core")
sys.path.append("../Core")


from Core.KeSyReSimple import classKeSyReSimple
#from Core import KeSyReSimple
#from KeSyReSimple import classKeSyReSimple



import matplotlib.pyplot as plt
import numpy as np


def f(t):
    #return np.exp(-t) * np.sin(2*np.pi*t)
    #return 3.5 * np.sin(2*t)
    #return 3.5 * np.sin(2*t) + 0.2 * np.sin(20*t)
    return np.exp(-0.2*t) * ( 3.5 * np.sin(2*t) + 0.2 * np.sin(20*t) )







if __name__ == '__main__':
  
  myKeSyReSimple = classKeSyReSimple()

  myKeSyReSimple.defineAdditiveFunctions(["sin"], "first")

  #myKeSyReSimple.defineNestedFunctions(["exp"], "first")

  myKeSyReSimple.defineAdditiveFunctions(["exp"], "second")

  myKeSyReSimple.defineMultiply(["first", "second"])


  myKeSyReSimple.setInputsDimension(1)


  myKeSyReSimple.Print()
  

  myKeSyReSimple.createModel()
  
  
  myKeSyReSimple.Print()
  
  
  print ("main: test with a simple function")


  myKeSyReSimple.compileModel()

  
  
  #
  # train
  # 
  
  x_axis = np.arange(0.0, 10.0, 0.101)
  
  X_train = x_axis
  Y_train = f(x_axis)

  x_axis_validation = np.arange(0.0, 10.0, 0.102)
  
  X_validation = x_axis_validation
  Y_validation = f(x_axis_validation)
  
  
  history = myKeSyReSimple._model.fit(
                       X_train,
                       Y_train,
                       validation_data = (X_validation,Y_validation),
                       epochs=1000,
                       verbose=0
                      )
  
  print ( history.history.keys() )
  
  


  myKeSyReSimple.PrintFormula()

 
  Y_predicted_validation = myKeSyReSimple._model.predict(X_validation)
  
  
  
  plt.plot(X_validation, Y_validation, "b .")
  plt.plot(X_validation, Y_predicted_validation, "r +")
  
  plt.show()
  
  
  



