# Importação dos módulos
import json

import cv2
import numpy as np


def set_control_region(rgb_image):
    """Permite a marcação das regiões de controle.

    Args:
        rgb_image: imagem de entrada.
    """

    # Converte a imagem para o padrão de cores apropriadas
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Cria uma cópia da imagem
    new_image = image.copy()

    # Cria uma janela para a imagem
    cv2.namedWindow('Image')
    cv2.imshow('Image', new_image)

    # Cria a lista para guardar a região de controle
    control_regions = set()

    # Define um mouse handler (para identificar os cliques)
    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if (y, x) not in control_regions:
                    # Guarda a coordenada clicada
                    control_regions.add((y, x))

                    # Desenha um círculo
                    cv2.circle(new_image, (x, y), 0, (0, 255, 127), -1)
                else:
                    # Remove a coordenada clicada
                    control_regions.remove((y, x))

                    # Recupera a cor original
                    color = tuple(image[y][x].tolist())

                    # Remove o círculo
                    cv2.circle(new_image, (x, y), 0, color, -1)

                cv2.imshow('Image', new_image)

    # Registra o mouse handler
    cv2.setMouseCallback('Image', mouse_click)

    # Espera o ESC do usuário
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Exibe as coordenadas das regiões marcadas
    print(control_regions)


def cluster_image(rgb_image, k, attempts):
    """Clusteriza a imagem por meio do Algoritmo KMeans.

    Args:
        rgb_image: imagem RGB de entrada.
        k: número de agrupamentos desejados.
        attempts: número de execuções do KMeans.

    Returns:
        array: rótulos determinados.
    """

    # Obtém uma representação vetorial da imagem RGB
    vectorized_im = rgb_image.reshape((-1, 3)).astype(np.float32)

    # Define o critério de parada para a execução do KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Indica que os centros iniciais devem ser obtidos aleatoriamente
    flag = cv2.KMEANS_RANDOM_CENTERS

    # Executa o KMeans e obtém o rótulo de cada elemento da imagem vetorizada
    _, labels, _ = cv2.kmeans(vectorized_im, k, None, criteria, attempts, flag)

    # Retorna o vetor com os rótulos em formato de linha
    return labels.flatten()


def segment_image(rgb_image, labels):
    """Realiza a segmentação da imagem com base nos rótulos.

    Args:
        rgb_image: imagem RGB de entrada.
        labels: array com os rótulos.
    """

    # Obtém uma representação vetorial da imagem RGB
    vectorized_im = rgb_image.reshape((-1, 3)).astype(np.float32)

    # Para cada rótulo possível
    for label in np.unique(labels):
        # Cria uma nova imagem de fundo branco
        new_image = np.full_like(
            vectorized_im, fill_value=255).astype(np.uint8)

        # Percorre todos os elementos da imagem vetorizada
        for idx in range(vectorized_im.shape[0]):
            # Verifica se o rótulo para a posição atual é igual ao rótulo de interesse
            if labels[idx] == label:
                # Se for, preenche essa posição com a cor respectiva da imagem vetorizada
                new_image[idx] = vectorized_im[idx]

        # Redimensiona a imagem para a representação 2D original
        new_image = new_image.reshape(rgb_image.shape)

        # Converte a imagem de RGB para BGR
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

        # Salva a imagem do segmento representado pelo rótulo
        cv2.imwrite(f'segmento_{label + 1}.png', new_image)


def compute_accuracy(original_image, segmented_image, control_regions):
    """Calcula a acurácia da segmentação da imagem por meio das regiões de controle.

    Args:
        original_image: matriz da imagem original.
        segmented_image: matriz da imagem segmentada.
        control_regions: array com os pontos das regiões de controle.

    Returns:
        float: acurácia calculada.
    """

    # Cria uma variável para guardar a taxa de acertos da segmentação
    accuracy = 0.

    # Para cada ponto das regiões de controle
    for x, y in control_regions:
        # Incrementa a acurácia caso as cores sejam iguais
        accuracy += (original_image[x][y] == segmented_image[x][y]).all()

    # Divide os acertos contados pela quantidade de pontos para se obter a taxa
    return accuracy / control_regions.shape[0]


def main():
    """Função principal do programa.
    """

    # Define uma seed para reprodutibilidade
    np.random.seed(13)

    # Faz a leitura da imagesm
    rgb_image = cv2.cvtColor(cv2.imread('./imagem.jpg'), cv2.COLOR_BGR2RGB)

    # Define as regiões de controle (uma única vez)
    # Clique com botão esquerdo + Ctrl -> Marca pixel
    # set_control_region(rgb_image)

    # Carrega na memória as regiões de controle definidas
    with open('./control_regions.json') as f:
        # Obtém as regiões de controle a partir do arquivo
        control_regions = json.load(f)['control_regions']

        # Converte as regiões para array
        control_regions = np.array(control_regions)

    # Realiza a clusterização dos componentes da imagem, obtendo-se os rótulos
    labels = cluster_image(rgb_image, k=3, attempts=5)

    # Realiza a segmentação dos componentes com base nos rótulos obtidos
    segment_image(rgb_image, labels)

    # Carrega a imagem com o segmento 3 (bolinhas azuis/violetas)
    segmented_image = cv2.imread('./segmento_3.png')
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    # Calcula a acurácia da segmentação por meio das regiões de controle
    acc = compute_accuracy(rgb_image, segmented_image, control_regions)

    # Exibe-se a taxa de acertos
    print(f'Taxa de Acertos: {acc * 100:05.2f}%')

    # Exibe-se a taxa de erros
    print(f'Taxa de Erros  : {(1 - acc) * 100:05.2f}%')


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a Função Principal
    main()
