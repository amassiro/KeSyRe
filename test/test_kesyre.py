
import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("Core")
sys.path.append("../Core")


from Core.KeSyRe import classKeSyRe
#from Core import KeSyRe
#from KeSyRe import classKeSyRe



import matplotlib.pyplot as plt
import numpy as np


def f(t):
    #return np.exp(-t) * np.sin(2*np.pi*t)
    #return 3.5 * np.sin(2*t)
    #return 3.5 * np.sin(2*t) + 0.2 * np.sin(20*t)
    return np.exp(-0.2*t) * ( 3.5 * np.sin(2*t) + 0.2 * np.sin(20*t) )







if __name__ == '__main__':
  
  myKeSyRe = classKeSyRe()

  myKeSyRe.defineAdditiveFunctions(["sin"], "first")

  #myKeSyRe.defineNestedFunctions(["exp"], "first")

  myKeSyRe.defineAdditiveFunctions(["exp"], "second")

  myKeSyRe.defineMultiply(["first", "second"])


  myKeSyRe.setInputsDimension(1)


  myKeSyRe.Print()
  

  myKeSyRe.createModel()
  
  
  myKeSyRe.Print()
  
  
  print ("main: test with a simple function")


  myKeSyRe.compileModel()

  
  
  #
  # train
  # 
  
  x_axis = np.arange(0.0, 10.0, 0.101)
  
  X_train = x_axis
  Y_train = f(x_axis)

  x_axis_validation = np.arange(0.0, 10.0, 0.102)
  
  X_validation = x_axis_validation
  Y_validation = f(x_axis_validation)
  
  #print ("X_train = ", X_train)
  #print ("Y_train = ", Y_train)
  
  
  history = myKeSyRe._model.fit(
                       X_train,
                       Y_train,
                       validation_data = (X_validation,Y_validation),
                       epochs=1000,
                       verbose=0
                      )
  
  print ( history.history.keys() )
  
  


  myKeSyRe.PrintFormula()

 
  Y_predicted_validation = myKeSyRe._model.predict(X_validation)
  
  
  
  plt.plot(X_validation, Y_validation, "b .")
  plt.plot(X_validation, Y_predicted_validation, "r +")
  
  plt.show()
  
  
  



