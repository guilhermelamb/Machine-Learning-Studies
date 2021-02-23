from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer


# =============================================================================
# Podemos definir vários parâmetros para a rede
# rede = buildNetwork(2, 3, 1, outclass = SoftmaxLayer,
#                     hiddenclass = SigmoidLayer, bias = True)
# 
# =============================================================================
rede = buildNetwork(2,3,1)

#Verifica qual a função de ativação da camada de entrada, oculta e saída
print(rede['in'])
print(rede['hidden0'])
print(rede['out'])


#Criar dataset (2 atributos previsores e uma classe)
base = SupervisedDataSet(2,1)
base.addSample((0,0), (0, ))
base.addSample((0,1), (1, ))
base.addSample((1,0), (1, ))
base.addSample((1,1), (0, ))

treinamento = BackpropTrainer(rede, dataset = base, learningrate=0.01,
                              momentum=0.06)

#Para controlar o números de épocas 
for i in range(1,30000):
    erro = treinamento.train()
    if i % 1000 == 0:
        print("Erro: %s" %erro)

#Pra ver qual resposta a rede vai retornar com esse input
print(rede.activate([0,0]))