# Importação dos módulos utilizados
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def add_noise(image, mu, sigma):
    """Adiciona Ruído Gaussiano a uma imagem em escala de cinza.

    Args:
        image: imagem em escala de cinza.

        mu: float que representa a média da função de densidade de
        probabilidade da gaussiana.

        sigma: float que representa o desvio padrão da função de
        densidade de probabilidade da gaussiana.

    Returns:
        imagem: imagem ruidosa.
    """

    # Cria a matriz de ruído com base na distribuição gaussiana
    noise = generate_noise(image, mu, sigma)

    # Salva a imagem do ruído gerado
    cv2.imwrite('./ruido.png', noise)

    # Cria o histograma do ruído (é possível verificar a gaussiana)
    plt.hist(noise.ravel(), 256, [0, 256])

    # Salva a imagem do histograma do ruído
    plt.savefig('./histograma_ruido.png')

    # Adiciona o ruído a imagem de interesse
    return cv2.add(image, noise)


def generate_noise(image, mu, sigma):
    """Gera a matriz com distribuição normal dos pixels que
    forma o ruído gaussiano.

    Args:
        image: imagem de interesse.

        mu: float que representa a média da função de densidade de
        probabilidade da gaussiana.

        sigma: float que representa o desvio padrão da função de
        densidade de probabilidade da gaussiana.

    Returns:
        matriz: matriz do ruído.
    """

    # Calcula o tamanho da matriz de ruídos de referência
    ref_size = 2048 ** 2

    # Representação unidimensional da matriz de referência
    ref_noise = np.zeros(shape=ref_size)

    # Atribui uma numeração para cada pixel da matriz
    # de ruídos de referência
    pixel_numbers = np.arange(start=0, stop=ref_size)

    # Embaralha o vetor com a numeração dos pixels
    np.random.shuffle(pixel_numbers)

    # Calcula o vetor da função de densidade de probabilidade
    # para cada intensidade de pixel
    probs = gaussian_pdf(np.arange(256), mu, sigma)

    # Inicia um contador para cada intensidade de pixel
    counter = np.zeros(256)

    # Índice inicial para o vetor com a numeração dos pixels
    curr_pxl = 0

    # Itera sobre todos os valores de intensidade de pixels para
    # uma imagem em escala de cinza
    for curr_int in range(256):

        # Calcula a quantidade de ocorrências da intensidade l
        # na matriz de ruído
        freq = np.rint(probs[curr_int] * ref_size)

        # Enquanto houverem pixels a serem preenchidos e quantidade
        # de pixels com intensidade l for menor que a frequência dela
        while curr_pxl < ref_size and counter[curr_int] + 1 <= freq:
            # Atribui a intensidade l no pixel da posição curr_pixel
            # do vetor de pixels numerados a matriz de referência
            ref_noise[pixel_numbers[curr_pxl]] = curr_int

            # Incrementa o contador para intensidade l
            counter[curr_int] += 1

            # Incrementa o índice do vetor dos pixels numerados
            curr_pxl += 1

    # Retorna a matriz do ruído gaussiano com as dimensões da imagem
    return ref_noise[0:image.size].reshape(*image.shape).astype(np.uint8)


def gaussian_pdf(z, mu, sigma):
    """Função de Densidade de Probabilidade da Gaussiana.

    Args:
        z: inteiro com um valor aleatório (nível de cinza).
        mu: float com a média da distribuição.
        sigma: float com o desvio padrão da distribuição.

    Returns:
        float: probabilidade/frequência de z.
    """

    # calcula a função de densidade de probabilidade para z
    p = (1 / (sigma * np.sqrt(2 * np.pi)))
    p *= np.exp(-1 * (((z - mu) ** 2) / (2 * (sigma ** 2))))

    # Retorna a probabilidade de z
    return p


def compute_metrics(image1, image2):
    """Calcula métricas de similaridade para comparar as imagens.

    Args:
        image1: imagem original.
        image2: imagem ruída.

    Returns:
        dict: dicionário com as métricas calculadas. A chave é o
        nome da métrica e o valor é a métrica em si.
    """

    # Converte o tipo das imagens para float32 com o
    # objetivo de evitar overflow
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Calcula o Erro Máximo (ME)
    max_error = compute_max_error(image1, image2)

    # Calcula o Erro Médio Absoluto (MAE)
    mae = compute_mae(image1, image2)

    # Calcula o Erro Médio Quadrático (MSE)
    mse = compute_mse(image1, image2)

    # Calcula a Raiz do Erro Médio Quadrático (RMSE)
    rmse = compute_rmse(image1, image2)

    # Calcula o Erro Médio Quadrático Normalizado (NMSE)
    nmse = compute_nmse(image1, image2)

    # Calcula o Coeficiente de Jaccard
    jaccard_coeff = compute_jaccard_coeff(image1, image2)

    # Cria um dicionário com as métricas
    metrics_dict = {
        'Erro Máximo (ME)': max_error,
        'Erro Médio Absoluto (MAE)': mae,
        'Erro Médio Quadrático (MSE)': mse,
        'Raiz do Erro Médio Quadrático (RMSE)': rmse,
        'Erro Médio Quadrático Normalizado (NMSE)': nmse,
        'Coeficiente de Jaccard': jaccard_coeff
    }

    # Retorna o dicionário das métricas
    return metrics_dict


def compute_max_error(image1, image2):
    """Calcula o Erro Máximo (ME).

    Args:
        image1: imagem original.
        image2: imagem ruidosa.

    Returns:
        float: valor da métrica.
    """

    # Retorna o valor da métrica
    return metrics.max_error(image1.flatten(), image2.flatten())


def compute_mae(image1, image2):
    """ Calcula o Erro Médio Absoluto (MAE).

    Args:
        image1: imagem original.
        image2: imagem ruidosa.

    Returns:
        float: valor da métrica.
    """

    # Retorna o valor da métrica
    return metrics.mean_absolute_error(image1.flatten(), image2.flatten())


def compute_mse(image1, image2):
    """Calcula o Erro Médio Quadrático (MSE).

    Args:
        image1: imagem original.
        image2: imagem ruidosa.

    Returns:
        float: valor da métrica.
    """

    # Retorna o valor da métrica
    return metrics.mean_squared_error(image1.flatten(), image2.flatten())


def compute_rmse(image1, image2):
    """Calcula o Raiz do Erro Médio Quadrático (RMSE).

    Args:
        image1: imagem original.
        image2: imagem ruidosa.

    Returns:
        float: valor da métrica.
    """

    # Representação unidimensional da matriz da imagem original
    flatten1 = image1.flatten()

    # Representação unidimensional da matriz da imagem ruidosa
    flatten2 = image2.flatten()

    # Retorna o valor da métrica
    return metrics.mean_squared_error(flatten1, flatten2, squared=False)


def compute_nmse(image1, image2):
    """Calcula o Erro Médio Quadrático Normalizado (NMSE).

    Args:
        image1: imagem original.
        image2: imagem ruidosa.

    Returns:
        float: valor da métrica.
    """

    # Calcula o numerador (MSE da imagem original com a ruidosa)
    p = compute_mse(image1, image2)

    # Calcula o denominador (MSE da imagem original com matriz nula)
    q = compute_mse(image1, np.zeros(image1.size))

    # Retorna o valor da métrica
    # 1e-10 é uma tolerância que evita a divisão por zero
    return p / (q + 1e-10)


def compute_jaccard_coeff(image1, image2):
    """Calcula o Coeficiente de Jaccard.

    Args:
        image1: imagem original.
        image2: imagem ruidosa.

    Returns:
        float: valor da métrica.
    """

    # Representação unidimensional da matriz da imagem original
    flatten1 = image1.flatten()

    # Representação unidimensional da matriz da imagem ruidosa
    flatten2 = image2.flatten()

    # Retorna o valor da métrica
    return metrics.jaccard_score(flatten1, flatten2, average='micro')


def main():
    """Função principal do programa
    """

    # Define uma seed para reprodutibilidade
    np.random.seed(13)

    # Carrega a imagem E
    image = cv2.imread('./imagem_e.png', cv2.IMREAD_GRAYSCALE)

    # Adiciona ruído a imagem
    # Ruído gaussiano é gerado a partir de uma distribuição
    # de densidade de probabilidades com média 128 e desvio 20
    noisy_image = add_noise(image, mu=80., sigma=20)

    # Salva a imagem ruidosa (degradada)
    cv2.imwrite('./imagem_e_degradada.png', noisy_image)

    # Calcula as métricas de similaridade entre a imagem
    # original e a imagem degradada
    metrics_dict = compute_metrics(image, noisy_image)

    # Exibe o nome e o valor de cada métrica computada
    for metric, value in metrics_dict.items():
        print(f'{metric} = {value}')


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a função principal
    main()
