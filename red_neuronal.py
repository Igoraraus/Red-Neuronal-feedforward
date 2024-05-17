import numpy as np
import random
import numpy as np
from PIL import Image



class red_neuronal(object):
    #tamaños contiene una lista con las capas y la cantida de neuronas por capa
    def __init__(self, tamaños):
        self.numero_capas = len(tamaños)
        self.tamaños = tamaños
        self.sesgos = [np.random.randn(y,1) for y in tamaños[1:]] #b
        self.pesos = [np.random.randn(y,x) for x, y in zip(tamaños[:-1],tamaños[1:])]#w
    #a' = sigmoide(wa+b)
    def feedforward(self, a): #evaluación de la red
        for b,w in zip(self.sesgos, self.pesos):
            a = sigmoide(np.dot(w,a)+b)
        return a
    
    def ddg(self, datos_de_entrenamiento, epochs, mini_batch_size,eta,test_data = None):
        if test_data: 
            test_data = list(test_data)
            n_test = len(test_data)
        n = len(datos_de_entrenamiento)
        for j in range(epochs):
            random.shuffle(datos_de_entrenamiento)
            mini_batches = [datos_de_entrenamiento[k:k+mini_batch_size] 
                            for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.actualizar_lote(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluacion(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
    def actualizar_lote(self, mini_batch, eta): #actualizacion de pesos y sesgos
        nabla_b = [np.zeros(b.shape) for b in self.sesgos]
        nabla_w = [np.zeros(w.shape) for w in self.pesos]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.retropropagacion(x,y)
            #calculo del gradiente, y luego se rellena la lista vacia de variaciones,
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #nb y nw son 0. luego se suman las variaciones, pero con una tasa de aprendizaje, para acelerar o disminuir.
        self.pesos = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.pesos, nabla_w)]
        self.sesgos = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.sesgos, nabla_b)]
    
    def retropropagacion(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.sesgos]
        nabla_w = [np.zeros(w.shape) for w in self.pesos]
        # feedforward
        activation = x #capa 1
        activations = [x] #matriz de todas las a de cada capa. irá aumentando al ejecutarse el loop
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.sesgos, self.pesos): #por capa l en L
            z = np.dot(w, activation)+b #calculo zl
            zs.append(z) #añado a zs
            activation = sigmoide(z) #la activación  al 
            activations.append(activation)
        # terminado este loop, se calcularon todas las activaciones, incluyendo la ultima capa
        # ahora, el paso siguiente es calcular el error de cada capa
        delta = self.derivadas_costo(activations[-1], y)*sigmoide_prima(zs[-1]) #paso 3
        nabla_b[-1] = delta 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        #retropropagación del error, para calcular los deltasw y deltasb
        for l in range(2, self.numero_capas):
            z = zs[-l]
            sp = sigmoide_prima(z)
            delta = np.dot(self.pesos[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    

    def evaluacion(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        #prueba para multiples resutlados

    def derivadas_costo(self, salidas, y):
        return (salidas-y)

    def identificar(self, x):
        # feedforward
        activation = x #capa 1
        activations = [x] #matriz de todas las a de cada capa. irá aumentando al ejecutarse el loop
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.sesgos, self.pesos): #por capa l en L
            z = np.dot(w, activation)+b #calculo zl
            zs.append(z) #añado a zs
            activation = sigmoide(z) #la activación  al
            activations.append(activation)
        return activations[-1]
#red.pesos[1] #matriz wij

def sigmoide(z):
    return 1.0/(1.0 + np.exp(-z))
#si se aplica a sigmoide un vector o matriz, numpy devolvera lo 
#mismo para cada elemento.
def sigmoide_prima(z):
    return sigmoide(z)*(1-sigmoide(z))


#red = red_neuronal([784,31,25,20,10])

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)


red = red_neuronal([784, 30, 10])
red.ddg(training_data, 5, 10, 6.0, test_data=test_data)

def save_image_from_pixels(data):
    pixel_list = data[0]
    
    # Reshape the 784 elements into a 28x28 array
    image_data = np.array(pixel_list).reshape(28, 28)

    # Create a new PIL Image object from the pixel array
    image = Image.fromarray((image_data * 255).astype(np.uint8), mode='L')

    # Save the image
    image.save('pixel_image.png')
def get_number_from_list(list_of_lists):
    for j, lst in enumerate(list_of_lists):
        if 1 in lst:
            return j
    return None

def numero(salida):
    j = -100
    for i in range(len(salida)):
        if salida[i][0] > j:
            j = salida[i][0]
            imax = i
    return print("el número es: "+str(imax))

from PIL import Image
import numpy as np

# Cargar la imagen como una matriz de profundidades normalizadas entre 0 y 1
image_path = "foto_3.jpg"
image = Image.open(image_path)

# Convertir la imagen a una matriz de profundidades
depth_matrix = np.array(image) / 255.0

# Redimensionar la matriz a un vector de largo 784 con listas de valores de profundidad
depth_vector = depth_matrix.reshape(-1, 1).tolist()
resultado = red.identificar(depth_vector)
print(numero(resultado))
print(resultado)
