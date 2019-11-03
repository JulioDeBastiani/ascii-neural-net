import matplotlib.image as mpimg
import os

#limpa arquivo
arquivo = open('EntrancesAndExits.txt', 'w')
arquivo.close()

directory = 'data/images'

#percorre diretorio com as imagens
for _, _, arquivo in os.walk(directory):
    filesVector = sorted(arquivo)

changePosition = 0
for fileName in filesVector:
    image = mpimg.imread(directory + '/' + fileName)
    
    binary = ''
    # loop por todos os pixeis
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if (image[x,y].all() == 0): #pixel eh preto
                binary += '1'
            else:
                binary += '0' #pixel nao eh preto

    bitPosition = ''
    for x in range(0, 47):
        if (changePosition == x):
            bitPosition += '1'
        else:
            bitPosition += '0'
        
    changePosition = changePosition + 1

    arquivo = open('EntrancesAndExits.txt', 'r')
    conteudo = arquivo.readlines()
    conteudo.append(fileName[0] + ' | ' + binary + ' | ' + bitPosition + '\n')
    arquivo = open('EntrancesAndExits.txt', 'w')
    arquivo.writelines(conteudo)
    arquivo.close()