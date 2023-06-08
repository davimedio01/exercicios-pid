# Importação dos módulos
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def convert_rgb2hsi(image_name, rgb_image):
    """Converte a matriz de uma imagem RGB para HSI.

    Args:
        image_name: nome do arquivo de imagem.
        rgb_image: imagem no espaço de cores RGB.

    Returns:
        imagem: imagem convertida para o espaço de cores HSI.
    """

    # Converte a imagem em sua representação matricial
    image_array = np.copy(np.asarray(rgb_image))

    # Itera na vertical
    for x in range(image_array.shape[0]):

        # Itera na horizontal
        for y in range(image_array.shape[1]):
            # Recupera e normaliza os valores das bandas R, G e B
            RGB = image_array[x][y] / 255

            # Chama a função de conversão de RGB para HSI
            HSI = compute_hsi(*RGB)

            # Atribui os valores de retorno de H, S e I na matriz da imagem
            image_array[x][y] = HSI

    # Cria uma imagem a partir da matriz
    hsi_image = Image.fromarray(image_array)

    # Salva a imagem no dispositivo de armazenamento
    hsi_image.save(f'{image_name}_hsi.png')

    # Retorna a imagem convertida
    return hsi_image


def compute_hsi(R, G, B):
    """ Calcula os valores de H, S e I equivalentes aos
    valores de entrada R, G e B.

    Args:
        R: inteiro com a intensidade de vermelho.
        G: inteiro com a intensidade de verde.
        B: inteiro com a intensidade de azul.

    Returns:
        array: array com os valores de H, S e I.
    """

    # Calcula o denominador da equação para obtenção de H
    denom = 2 * np.sqrt(((R - G) ** 2) + (R - B) * (G - B))

    # Calcula a componente H
    H = np.arccos(((R - G) + (R - B)) / (denom + 1e-8))

    # Se o valor B for maior que G
    if B > G:
        # Limita o valor de H no intervalo [0, 360]
        H = 360 - H

    # Se o valor mínimo entre R, G e B for 1/3
    if min(R, G, B) == 1 / 3:
        # Atribui valor zero para S
        S = 0
    # Caso o valor mínimo entre R, G e B for zero
    elif min(R, G, B) == 0.:
        # Atribui valor 1 para S
        S = 1
    # Caso contrário
    else:
        # O Valor de S é dado pela equação
        S = 1 - (3 * min(R, G, B) / (R + G + B + 1e-8))

    # Calcula o valor de I
    I = (R + G + B) / 3

    # Retorna a conversão dos valores R, G e B para H, S e I
    # Para o módulo Pillow, é necessário que:
    # H seja entre 0 e 180
    # S e I sejam entre 0 e 255
    return np.array([H / 2, S * 255, I * 255])


def save_image_bands(image_name, hsi_image):
    """ Separa as bandas de uma imagem HSI e salva
    a imagem correspondente a cada.

    Args:
        image_name: string com o nome da imagem.
        image: imagem de entrada.
    """

    # Converte a imagem em uma matrix
    image_array = np.asarray(hsi_image)

    # Separa as matrizes correspondentes a cada banda
    H = image_array[:, :, 0]
    S = image_array[:, :, 1]
    I = image_array[:, :, 2]

    # Cria a imagem da componente H (em escala de cinza)
    H_image = Image.fromarray(H, mode='L')

    # Cria a imagem da componente S (em escala de cinza)
    S_image = Image.fromarray(S, mode='L')

    # Cria a imagem da componente I (em escala de cinza)
    I_image = Image.fromarray(I, mode='L')

    # Salva a imagem da banda H
    H_image.save(f'{image_name}_h.png')

    # Salva a imagem da banda S
    S_image.save(f'{image_name}_s.png')

    # Salva a imagem da banda I
    I_image.save(f'{image_name}_i.png')


def create_histogram(image_name, hsi_image):
    """Gera o histograma de uma imagem HSI.

    Args:
        image_name: string com o nome da imagem.
        hsi_image: imagem de entrada.
    """

    # Separa a componente I da imagem
    I = np.asarray(hsi_image)[:, :, 2]

    # Cria o histograma da imagem com base em I
    plt.hist(I.flatten(), 256, [0, 256], color='#1f77b4')

    # Salva o histograma
    plt.savefig(f'hist_{image_name}.png')

    # Fecha a instância da figura
    plt.close()


def equalize_histogram(image_name, hsi_image):
    """Equaliza o histograma de uma imagem HSI.

    Args:
        image_name: string com o nome da imagem.
        hsi_image: imagem de entrada.

    Returns:
        imagem: imagem equalizada.
    """

    # Cria uma cópia da matrix da imagem HSI
    hsi_image_array = np.copy(np.asarray(hsi_image))

    # Separa a componente I da imagem
    I = hsi_image_array[:, :, 2]

    # Cria uma representação unidimensional de I
    flatten_I = I.flatten()

    # Pega os níveis de intensidade em I e suas frequências
    values, freqs = np.unique(flatten_I, return_counts=True)

    # Calcula a Função de Probabilidade Acumulada
    fda = np.cumsum(freqs / flatten_I.size)

    # Calcula os novos níveis de intensidade
    new_values = np.rint(fda * np.max(values)).astype(np.uint8)

    # Mapeia os níveis de intensidade originais com os novos
    mapping = dict(zip(values, new_values))

    # Cria a nova componente I a partir do mapeamento
    new_I = np.array([mapping[v] for v in flatten_I])

    # Redimensiona a componente
    new_I = new_I.reshape(*np.asarray(I).shape)

    # Atualiza a componente I da imagem
    hsi_image_array[:, :, 2] = new_I

    # Retorna a imagem equalizada
    return Image.fromarray(hsi_image_array)


def main():
    """Função principal do programa
    """

    # Carrega as imagens e armazena-as no dicionário
    images_dict = {
        'Img1': Image.open('./Img1.bmp').convert('RGB'),
        'Img2': Image.open('./Img2.bmp').convert('RGB'),
        'Img3': Image.open('./Img3.bmp').convert('RGB'),
    }

    # Passa por cada imagem no dicionário
    for image_name, image in images_dict.items():
        # Converte imagem de RGB para HSI
        hsi_image = convert_rgb2hsi(image_name, image)

        # Salva cada uma das componentes da imagem HSI
        save_image_bands(image_name, hsi_image)

        # Cria o histograma da imagem HSI
        create_histogram(image_name, hsi_image)

        # Realiza a equalização do histograma da imagem HSI
        equalized_image = equalize_histogram(image_name, hsi_image)

        # Salva a imagem HSI equalizada
        equalized_image.save(f'{image_name}_equalizada.png')

        # Cria o histograma da imagem HSI equalizada
        create_histogram(f'{image_name}_equalizada', equalized_image)

        # Separa a componente I equalizada
        equalized_I = np.asarray(equalized_image)[:, :, 2]

        # Cria uma imagem da componente I equalizada
        equalized_I = Image.fromarray(equalized_I, mode='L')

        # Salva a imagem
        equalized_I.save(f'{image_name}_i_equalizado.png')


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a função principal
    main()
