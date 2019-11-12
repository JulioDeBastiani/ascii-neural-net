import matplotlib.image as mpimg
import random
import os

def adicionaRuido(firstRange, secondRange):
    content = []
    for _ in range(firstRange):
        line = random.choice(trainingDataSet)
        a = list(range(0,48))
        random.shuffle(a)
        for x in range(secondRange):
            position = a[x]
            if (int(line[position])):
                line = change_char(line, position, '0')
            else:
                line = change_char(line, position, '1')
        content.append(line)
    
    return content

def change_char(s, p, r):
    return s[:p]+r+s[p+1:]

#limpa arquivo
arquivo = open('testDataset.txt', 'w')
arquivo.close()

trainingDataSet = open('EntrancesAndExits.txt').read().splitlines()

content = []
binaryVec = []

#caracteres do dataset de treino
for i in range(34):
    line = random.choice(trainingDataSet)
    binaryVec.append(line[48:])
    content.append(line[0:48])

#ruido minimo (muda 2 bits)
for line in adicionaRuido(20, 2):
    binaryVec.append(line[48:])
    content.append(line[0:48])

#ruido medio (muda 6 bits)
i = 0
for line in adicionaRuido(20, 6):
    binaryVec.append(line[48:])
    content.append(line[0:48])

#ruido avancado (muda 12 bits)
for line in adicionaRuido(20, 12):
    binaryVec.append(line[48:])
    content.append(line[0:48])

#imagens que nao fazem parte do conjunto de treinamento
directory = 'data/images-to-not-test'

for _, _, arquivo in os.walk(directory):
    filesVector = sorted(arquivo)

for x in range(0,6):
    fileName = filesVector[random.randint(0, 1)]
    image = mpimg.imread(directory + '/' + fileName)
    binary = ''
    # loop por todos os pixeis
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if (image[x,y].all() == 0): #pixel eh preto
                binary += '1'
            else:
                binary += '0' #pixel nao eh preto
    content.append(binary)
    binaryVec.append(' 000000000000000000000000000000000000')

i = 0
for line in content:
    arquivo = open('testDataset.txt', 'r')
    newContent = arquivo.readlines()
    if (i < len(binaryVec)):
        newContent.append(line + binaryVec[i] + '\n')
    arquivo = open('testDataset.txt', 'w')
    arquivo.writelines(newContent)
    arquivo.close()
    i = i + 1