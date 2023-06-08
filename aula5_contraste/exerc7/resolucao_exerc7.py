# Importação dos módulos
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def equalize_histogram(image_name, image):
    """Equaliza o histograma da imagem de entrada.

    Args:
        image_name: string com o nome da imagem.
        image: objeto da imagem de entrada.
    """

    # Cria o histograma da imagem de entrada
    create_histogram(f'hist_{image_name}.png', np.asarray(image))

    # Cria uma representação unidimensional da imagem
    flatten_image = np.asarray(image).flatten()

    # Pega os níveis de intensidade da imagem e suas frequências
    values, freqs = np.unique(flatten_image, return_counts=True)

    # Calcula a Função de Probabilidade Acumulada
    fda = np.cumsum(freqs / flatten_image.size)

    # Calcula os novos níveis de intensidade
    new_values = np.rint(fda * np.max(values)).astype(np.uint8)

    # Mapeia os níveis de intensidade originais com os novos
    mapping = dict(zip(values, new_values))

    # Cria a imagem equalizada a partir do mapeamento
    new_image = np.array([mapping[v] for v in flatten_image])

    # Redimensiona a imagem
    new_image = new_image.reshape(*np.asarray(image).shape)

    # Cria o histograma da nova imagem
    create_histogram(f'hist_{image_name}_equalizada.png', new_image)

    print('Histogramas com e sem equalização salvos.')

    # Cria o objeto da imagem equalizada a partir de sua matrix
    new_image = Image.fromarray(new_image.astype(np.uint8), mode='L')

    # Salva a imagem equalizada
    new_image.save(f'{image_name}_equalizada.png')

    print('Imagem equalizada gerada com sucesso.')


def create_histogram(filename, image):
    """Gera o histograma da imagem de entrada.

    Args:
        filename: string com o nome da imagem.
        image: objeto da imagem de entrada.
    """

    # Cria o histograma do imagem
    plt.hist(image.ravel(), 256, [0, 256], color='#1f77b4')

    # Salva o histograma
    plt.savefig(filename)

    # Fecha a instância da figura
    plt.close()


def main():
    """Função principal do programa
    """

    # Cria o dicionário com o nome e o objeto das imagens
    images_dict = {
        'frutas': Image.open('./frutas.png'),
        'mammogram': Image.open('./mammogram.png'),
        'moon': Image.open('./moon.png'),
        'polem': Image.open('./polem.png')
    }

    # Itera as tuplas do dicionário
    for image_name, image in images_dict.items():
        # Exibe o nome da imagem
        print(f'\nIMAGEM {image_name}')

        # Equaliza o histograma da imagem
        equalize_histogram(image_name, image.convert('L'))


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a função principal
    main()
