import math

import numpy as np
from PIL import Image, ImageColor


class HKAlgorithm:
    """Adapta o Algoritmo de Hoshen Kopelman para
    imagens em níveis dede cinza (grayscale).
    """

    def __init__(self, image):
        """Inicializa a classe HKAlgorithm.

        Args:
            image: imagem de interesse.
        """

        # Converte a imagem em uma matriz
        self.matrix = np.asarray(image)

        # Cria uma lista com a quantidade máxima possível de rótulos
        self.labels = np.arange(0, self.matrix.size + 1, dtype=int)

    def label(self):
        """Performa a rotulagem da imagem, separando-a em clusters.

        Returns:
            inteiro: quantidade de clusters presentes na imagem.
        """

        # Inicializa o contador de rótulos
        curr_label = 0

        # Cria uma matriz de zeros com as dimensões da imagem para
        # atribuição dos rótulos
        labeled_matrix = np.zeros_like(self.matrix, dtype=int)

        # Itera as linhas da imagem
        for x in range(self.matrix.shape[0]):

            # Itera as colunas da imagem
            for y in range(self.matrix.shape[1]):

                # Inicializa a lista dos rótulos vizinhos
                # da posição (x, y), considerando vizinhança-8
                neighbors = [0, 0, 0, 0]

                # Se o vizinho da esquerda estiver nos limites da imagem
                if x - 1 >= 0:
                    # Atribui a primeira posição da lista seu rótulo
                    neighbors[0] = labeled_matrix[x - 1, y]

                # Se o vizinho de cima estiver nos limites da imagem
                if y - 1 >= 0:
                    # Atribui a segunda posição da lista seu rótulo
                    neighbors[1] = labeled_matrix[x, y - 1]

                # Se o vizinho da diagonal suérior esquerda estiver nos
                # limites da imagem
                if x - 1 >= 0 and y - 1 >= 0:
                    # Atribui a terceira posição da lista seu rótulo
                    neighbors[2] = labeled_matrix[x - 1, y - 1]

                # Se o vizinho da diagonal superior direita estiver nos
                # limites da imagem
                if x - 1 >= 0 and y + 1 < self.matrix.shape[1]:
                    # Atribui a última posição da lista seu rótulo
                    neighbors[3] = labeled_matrix[x - 1, y + 1]

                # Cria uma lista de tuplas na qual cada posição armazena
                # o rótulo e a posição do vizinho
                neighbors = list(zip(neighbors, list(range(0, 4))))

                # Lista com as coordenadas de cada vizinho
                neighbors_coords = [
                    (x-1, y), (x, y-1), (x-1, y-1), (x-1, y+1)
                ]

                # Cria uma lista apenas com os vizinhos já rotulados
                labeled_neighbors = [n for n in neighbors if n[0] != 0]

                # Se o labeled_neighbors for maior que zero
                if len(labeled_neighbors) > 0:
                    # Inicializa a lista que armazena os rótulos
                    # dos vizinhos que fazem parte do mesmo cluster
                    cluster = []

                    # Para cada rótulo dos vizinhos rotulados
                    for n_label, idx in labeled_neighbors:
                        # Recuperamos a coordenada do vizinho i de
                        # rótulo n com posição idx
                        x1, y1 = neighbors_coords[idx]

                        # Se a intensidade do pixel do vizinho i for
                        # igual a da posição atual da imagem
                        if self.matrix[x][y] == self.matrix[x1][y1]:
                            # O vizinho é inserido em "clusters"
                            cluster.append(n_label)

                    # Se o tamanho de cluster for maior que zero
                    if len(cluster) > 0:
                        # Guarda o rótulo do vizinho com menor rótulo
                        min_neighbor = min(cluster, key=lambda n: n)

                        # Para cada rótulo em clusters
                        for n_label in cluster:
                            # Se o rótulo n for diferente do menor
                            if n_label != min_neighbor:
                                # Realiza a união entre os rótulos,
                                # indicando que o rótulo n faz parte
                                # do agrupamento do menor rótulo
                                self.uf_merge(n_label, min_neighbor)

                        # Atribui o menor rótulo a posição (x, y)
                        labeled_matrix[x][y] = min_neighbor
                    else:
                        # Caso contrário, um novo cluster é encontrado
                        # Incrementa-se a quantidade de rótulos
                        curr_label += 1

                        # Atribui o novo rótulo a posição (x, y)
                        labeled_matrix[x][y] = curr_label
                else:
                    # Caso contrário, um novo cluster é encontrado
                    # Incrementa-se a quantidade de rótulos
                    curr_label += 1

                    # Atribui o novo rótulo a posição (x, y)
                    labeled_matrix[x][y] = curr_label

        # Efetua a rerotulagem e retorna a matriz com os rótulos
        return self._relabel(labeled_matrix)

    def uf_find(self, x):
        """Encontra o rótulo representante do rótulo x.

        Args:
            x: inteiro que indica o rótulo de um cluster.

        Returns:
            inteiro: o rótulo que representa o cluster.
        """

        # Guarda o valor do rótulo x
        root = x

        # Enquanto o rótulo representante do cluster for diferente
        # do rótulo raiz
        while self.labels[root] != root:
            # Atribui o rótulo representante à variável root
            root = self.labels[root]

        # Enquanto o rótulo representante do cluster for diferente
        # do rótulo raiz x
        while self.labels[x] != x:
            # Altera o rótulo representante de x para root
            # Atribui o rótulo representante à variável x
            x, self.labels[x] = self.labels[x], root

        # Retorna o rótulo representante do cluster
        return root

    def uf_merge(self, x, y):
        """Realiza a união entre os cluster com rótulo x e y.

        Args:
            x: inteiro representando o rótulo do primeiro cluster.
            y: inteiro representando o rótulo do segundo cluster.
        """

        # Relaciona os rótulos x e y na list de rótulos.
        self.labels[self.uf_find(x)] = self.uf_find(y)

    def _relabel(self, labeled_matrix):
        """Performa a rotulação novamente para garantir
        que os rótulos dos cluster estão corretos.

        Args:
            labeled_matrix: matrix com os agrupamentos rotulados.

        Returns:
            inteiro: quantidade de rótulos da imagem.
            matriz: matriz com rótulos atualizados.
        """

        # Cria uma lista de zeros com a quantidade de rótulos contados
        # no método label
        new_labels = np.zeros_like(self.labels, dtype=np.int8)

        # Inicializa o contador de rótulos
        curr_label = 0

        # Itera as linhas da imagem
        for x in range(self.matrix.shape[0]):

            # Itera as colunas da imagem
            for y in range(self.matrix.shape[1]):

                # Se a posição (x, y) possui rótulo
                if labeled_matrix[x][y]:
                    # Encontra o rótulo representante do rótulo raiz
                    root_label = self.uf_find(labeled_matrix[x][y])

                    # Se o rótulo raiz não possui representante
                    if not new_labels[root_label]:
                        # Um novo cluster é encontrado
                        # Incrementa-se a quantidade de rótulos
                        curr_label += 1

                        # Atribui o novo rótulo como representante
                        # do rótulo raiz
                        new_labels[root_label] = curr_label

                    # Atribui o rótulo representante a posição (x, y)
                    labeled_matrix[x][y] = new_labels[root_label]

        # Retorna a quantidade de rótulo e a matriz rotulada
        return curr_label, labeled_matrix


def find_clusters_distance(labels, labeled_matrix):
    """Calcula a distância entre dois agrupamentos tomando como
    referência seus pontos centrais (centros).

    Args:
        labels: array de inteiros na qual cada elemento indica o
        rótulo de um agrupamento.

        labeled_matrix: matrix com os agrupamentos rotulados.

    Returns:
        tupla: tupla de floats com as distâncias Euclidiana, de
        Manhattan e de Chebyshev entre os centros dos agrupamentos.
    """

    # Determina o centro do primeiro agrupamento
    cluster_center1 = find_cluster_center(labels[0], labeled_matrix)

    # Determina o centro do segundo agrupamento
    cluster_center2 = find_cluster_center(labels[1], labeled_matrix)

    # Calcula as distâncias Euclidiana, de Manhattan e de Chebyshev
    # entre os centros dos agrupamentos
    de = euclidean_distance(cluster_center1, cluster_center2)
    d4 = cityblock_distance(cluster_center1, cluster_center2)
    d8 = chessboard_distance(cluster_center1, cluster_center2)

    # Retorna as métricas
    return de, d4, d8


def find_cluster_center(label, labeled_matrix):
    """Calcula o centro de um agrupamento presente na matrix de rótulos.

    Args:
        label: inteiro que indica o rótulo do agrupamento.
        labeled_matrix: matrix com os agrupamentos rotulados.

    Returns:
        tupla: tupla com as coordenadas do centro do agrupamento.
    """

    # Separa as coordenadas da matrix de rótulos referentes
    # ao rótulo do agrupamento de interesse
    cluster_indices = np.argwhere(labeled_matrix == label)

    # Calcula o ponto central por meio de uma média aritmética
    # das coordenadas do agrupamento
    return tuple(cluster_indices.mean(axis=0))


def euclidean_distance(point1, point2):
    """Calcula a Distância Euclidiana entre dois pontos.

    Args:
        point1: tupla com as coordenadas do ponto 1.
        point2: tupla com as coordenadas do ponto 2.

    Returns:
        float: distância entre os dois pontos.
    """

    # Separa as coordenadas do ponto 1
    x1, y1 = point1

    # Separa as coordenadas do ponto 2
    x2, y2 = point2

    # Calcula a métrica
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


def cityblock_distance(point1, point2):
    """Calcula a Distância de Manhattan entre dois pontos.

    Args:
        point1: tupla com as coordenadas do ponto 1.
        point2: tupla com as coordenadas do ponto 2.

    Returns:
        float: distância entre os dois pontos.
    """

    # Separa as coordenadas do ponto 1
    x1, y1 = point1

    # Separa as coordenadas do ponto 2
    x2, y2 = point2

    # Calcula a métrica
    return abs(x1 - x2) + abs(y1 - y2)


def chessboard_distance(point1, point2):
    """Calcula a Distância de Chebyshev entre dois pontos.

    Args:
        point1: tupla com as coordenadas do ponto 1.
        point2: tupla com as coordenadas do ponto 2.

    Returns:
        float: distância entre os dois pontos.
    """

    # Separa as coordenadas do ponto 1
    x1, y1 = point1

    # Separa as coordenadas do ponto 2
    x2, y2 = point2

    # Calcula a métrica
    return max(abs(x1 - x2), abs(y1 - y2))


def choose_labels(max_label):
    """Escolhe aleatoriamente, sem repetição, dois rótulos inteiros
    de 1 até um certo limite (max_label).

    Args:
        max_label: inteiro que indica o maior rótulo possível.

    Returns:
        array: array de inteiros com os dois rótulos selecionados.
    """

    # Gera um array com todos os possíveis rótulos de 1 até max_label
    possible_labels = np.arange(1, max_label + 1)

    # Seleciona dois rótulos do array de forma aleatória
    return np.random.choice(possible_labels, size=2, replace=False)


def color_clusters(image_name, labeled_matrix, colors):
    """Colore as regiões rotuladas para evidenciar os clusters.

    Args:
        image_name: string com o nome da imagem.
        labeled_matrix: matrix com os agrupamentos rotulados.
        colors: lista com o nome das cores.
    """

    # Cria uma matriz 3D, representando as 3 bandas para uma imagem RGB
    clusters = np.zeros(shape=(*labeled_matrix.shape, 3), dtype=int)

    # Itera nas linhas da imagem
    for x in range(clusters.shape[0]):

        # Itera nas colunas da imagem
        for y in range(clusters.shape[1]):

            # Atribui a cor relacionada ao rótulo x
            curr_color = colors[labeled_matrix[x][y] - 1]

            # Busca os valores RGB da cor
            curr_color = ImageColor.getrgb(curr_color)

            # Itera nas bandas da imagem
            for z in range(clusters.shape[2]):
                # Atribui a intensidade de pixel na posição (x, y)
                # para a cor escolhida
                clusters[x][y][z] = curr_color[z]

    # Salva um objeto imagem RGB
    image = Image.fromarray(clusters.astype(np.uint8), mode='RGB')
    image.save(f'clusters_{image_name.lower()}.bmp')


def main():
    """Função principal do programa
    """

    # Define uma seed para reprodutibilidade
    np.random.seed(13)

    # PRIMEIRA PARTE DO EXERCÍCIO 13
    print('ROTULAGEM DAS IMAGENS VIA ALGORITMO HOSHEN-KOPELMAN')

    # Cria um dicionário para carregar as imagens
    images_dict = {
        'A': Image.open('./imagem_a.bmp'),
        'B': Image.open('./imagem_b.bmp'),
        'C': Image.open('./imagem_c.bmp'),
        'D': Image.open('./imagem_d.bmp'),
        'E': Image.open('./imagem_e.bmp'),
    }

    # Cria um vetor de cores para representar cada cluster
    colors = ['red', 'green', 'blue', 'yellow',
              'magenta', 'cyan', 'orange', 'lime',
              'purple', 'teal', 'pink', 'brown',
              'navy', 'silver', 'gold', 'gray']

    # Identifica as imagens e exibe suas propriedades
    for image_name, image in images_dict.items():
        # Exibe o nome da imagem
        print(f'\nIMAGEM {image_name}')

        # Cria uma instância da classe HKAlgorithm para a imagem atual
        hk_algorithm = HKAlgorithm(image=image)

        # Rotula a imagem atual
        label_count, labeled_matrix = hk_algorithm.label()

        # Exibe o total de componentes conexos encontrados
        print(f'Total de Componentes Conexos = {label_count}')

        # Exibe a matrix com os rótulos atribuídos a cada pixel
        print('Rótulos Atribuídos a Imagem:')
        print(labeled_matrix)

        # Cria uma representação visual dos clusters da imagem
        color_clusters(image_name, labeled_matrix, colors)

    # SEGUNDA PARTE DO EXERCÍCIO 13
    print('\n\nCÁLCULO DAS DISTÂNCIAS ENTRE CLUSTERS DA IMAGEM E')

    # Exibe o número de componentes conexos na imagem E
    print(f'Total de Componentes Conexos = {label_count}')

    # Seleciona dois rótulos aleatórios da Imagem E
    chosen_labels = choose_labels(label_count)

    # Exibe os rótulos selecionados
    print(f'\nClusters Selecionados = {chosen_labels}')

    # Calcula distâncias entre os clusters formados pelos rótulos
    de, d4, d8 = find_clusters_distance(chosen_labels, labeled_matrix)

    # Exibe as distâncias calculadas
    print(f'\nDistância Euclidiana (DE) = {de}')
    print(f'Distância Cityblock (D4) = {d4}')
    print(f'Distância Chessboard (D8) = {d8}')


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a função principal
    main()
