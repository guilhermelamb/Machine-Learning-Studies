#Exemplo de construção de rede manual

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

#Instanciando a rede
rede = FeedForwardNetwork()

#Definindo a quantidade de neurônios por camada
camadaEntrada = LinearLayer(2)
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)

#Criando unidades de bias
bias1 = BiasUnit()
bias2 = BiasUnit()

#Adicionar as camadas e bias na rede
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

#Realizar a ligação entre as camadas
entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

#Efetivamente construir a rede
rede.sortModules()

#Print das informações da rede
print(rede)

#Mostra os pesos aleatórios que foram gerados
print(entradaOculta.params)
print(ocultaSaida.params)
print(biasOculta.params)
print(biasSaida.params)
