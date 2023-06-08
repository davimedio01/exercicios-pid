# Importação dos módulos
import numpy as np
from PIL import Image
from sklearn import metrics


def apply_gamma_correction(image, gamma, c=1, epsilon=1e-10):
    """Aplica a correção gama na imagem de entrada. Considera
    c = 1 e epsilon = 1e-10 como valores padrões.

    Args:
        image: objeto da imagem de entrada.
        c: constante positiva que desloca os valores da correção.
        gamma: constante positiva associada a correção.
        epsilon: compensação para quando o pixel de entrada for zero.
    """

    # Obtém a matrix que representa a imagem
    image = np.asarray(image)

    # Conver a matrix em um vetor unidimensional
    flatten_image = image.ravel()

    # Aplica a transformação gama ao vetor
    # Aqui, divide-se o valor para garantir valores entre 0 e 1
    # Depois, leva os valores obtidos para a faixa de 0 a 255 novamente
    new_image = c * np.power((flatten_image + epsilon) / 255, gamma) * 255

    # Caso haja algum valor fora do intervalo, aproxima-o para os limites
    # Converte o vetor unidimensional de volta em matrix
    new_image = np.clip(new_image, 0, 255).reshape(*image.shape)

    # Transforma a matrix em um objeto do tipo imagem
    new_image = Image.fromarray(new_image.astype(np.uint8), mode='L')

    # Salva a nova imagem
    new_image.save(f'imagem_corrigida_gama_{gamma}.png')


def compute_metrics(image1, image2):
    """Calcula métricas de similaridade para comparar as imagens.

    Args:
        image1: imagem original.
        image2: imagem a ser comparada.

    Returns:
        dict: dicionário com as métricas calculadas. A chave é o
        nome da métrica e o valor é a métrica em si.
    """

    # Converte o tipo das imagens para float32 com o
    # objetivo de evitar overflow
    image1 = np.asarray(image1).astype(np.float32)
    image2 = np.asarray(image2).astype(np.float32)

    # Calcula o Erro Médio Absoluto (MAE)
    mae = compute_mae(image1, image2)

    # Calcula o Erro Médio Quadrático (MSE)
    mse = compute_mse(image1, image2)

    # Cria um dicionário com as métricas
    metrics_dict = {
        'Erro Médio Absoluto (MAE)': mae,
        'Erro Médio Quadrático (MSE)': mse
    }

    # Retorna o dicionário das métricas
    return metrics_dict


def compute_mae(image1, image2):
    """ Calcula o Erro Médio Absoluto (MAE).

    Args:
        image1: imagem original.
        image2: imagem a ser comparada.

    Returns:
        float: valor da métrica.
    """

    # Retorna o valor da métrica
    return metrics.mean_absolute_error(image1.ravel(), image2.ravel())


def compute_mse(image1, image2):
    """Calcula o Erro Médio Quadrático (MSE).

    Args:
        image1: imagem original.
        image2: imagem a ser comparada.

    Returns:
        float: valor da métrica.
    """

    # Retorna o valor da métrica
    return metrics.mean_squared_error(image1.ravel(), image2.ravel())


def main():
    """Função principal do programa
    """

    # Carrega a imagem E
    image = Image.open('./imagem_e.png')

    # Carrega a imagem E degrada com ruído gaussiano (Aula4 - Ex. 4)
    noisy_image = Image.open('./imagem_e_degradada.png')

    # Aplica a correção para gama = 0.04
    apply_gamma_correction(noisy_image, gamma=0.04)

    # Aplica a correção para gama = 0.4
    apply_gamma_correction(noisy_image, gamma=0.4)

    # Aplica a correção para gama = 2.5
    apply_gamma_correction(noisy_image, gamma=2.5)

    # Aplica a correção para gama = 10
    apply_gamma_correction(noisy_image, gamma=10)

    # Carrega as imagens corrigidas
    corrected_images_dict = {
        '0.04': Image.open('./imagem_corrigida_gama_0.04.png'),
        '0.4': Image.open('./imagem_corrigida_gama_0.4.png'),
        '2.5': Image.open('./imagem_corrigida_gama_2.5.png'),
        '10': Image.open('./imagem_corrigida_gama_10.png')
    }

    # Itera o dicionário, recendo a correção e a imagem corrigida
    for gamma, corrected_image in corrected_images_dict.items():
        print(f'\nIMAGEM COM CORREÇÃO GAMA = {gamma}')

        print('\nComparação entre imagem original e imagem degradada')

        # Calcula MAE e MSE entre imagem E e imagem E degradada
        # por ruído gaussiano
        metrics_dict_noisy = compute_metrics(image, noisy_image)

        # Exibe o nome e o valor das métricas computadas
        for metric, value in metrics_dict_noisy.items():
            print(f'{metric} = {value}')

        print('\nComparação entre imagem original e imagem corrigida')

        # Calcula MAE e MSE entre imagem E e imagem E corrigida
        # por meio da correção gama
        metrics_dict_corrected = compute_metrics(image, corrected_image)

        # Exibe o nome e o valor das métricas computadas
        for metric, value in metrics_dict_corrected.items():
            print(f'{metric} = {value}')

        print('\n----------')


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a função principal
    main()
