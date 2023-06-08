import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_image_a():
    """Cria uma imagem A em escala de cinza com dimensão 5x5.
    """

    # Cria um vetor com 25 níveis de cinza escolhidos aleatoriamente
    flatten_image = np.random.choice(list(range(0, 256)), size=25)

    # Transforma o vetor em uma matriz 5x5
    image = flatten_image.reshape(5, 5)

    # Converte para um objeto imagem
    image = Image.fromarray(image.astype(np.uint8), mode='L')

    # Salva a imagem gerada
    image.save('./imagem_a.png')


def add_gaussian_noise(image, mu, sigma):
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
    noise = generate_gaussian_noise(image, mu, sigma)

    # Salva a imagem do ruído gerado
    cv2.imwrite('./ruido_gaussiano.png', noise)

    print('Imagem do ruído gaussiano salva.')

    # Cria o histograma do ruído
    create_histogram('./hist_ruido_gaussiano.png', noise)

    print('Histograma do ruído gaussiano salvo.')

    # Adiciona o ruído a imagem de interesse
    noisy_image = cv2.add(image, noise)

    # Salva a imagem ruidosa (degradada)
    # IMAGEM A degradada com ruído (IMAGEM B)
    cv2.imwrite('./imagem_a_ruido_gaussiano.png', noisy_image)

    print('Imagem ruidosa salva com sucesso.')


def generate_gaussian_noise(image, mu, sigma):
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

    # Exibe o vetor de p(z)
    print('Resultado da Função p(z)')
    print(probs)

    # Normaliza o vetor das probabilidades para garantir
    # que a soma seja 1, o que garante que a matriz de
    # ruído seja totalmente preenchida
    probs /= np.sum(probs)

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


def apply_sp_noise(image, salt_prob, pepper_prob):
    """Aplica Ruído Sal e Piemta a uma imagem em escala de cinza.

    Args:
        image: imagem em escala de cinza.

        salt_prob: float que representa a porcentagem de pixels
        com intensidade clara.

        pepper_prob: float que representa a porcentagem de pixels
        com intensidade escura.

    Returns:
        imagem: imagem ruidosa.
    """

    # Cria a matriz de ruído com base nas probabilidades de sal e pimenta
    noise = generate_sp_noise(image, salt_prob, pepper_prob)

    # Salva a imagem do ruído gerado
    cv2.imwrite('./ruido_salpimenta.png', noise)

    print('Imagem do ruído sal e pimenta salva.')

    # Adiciona o ruído a imagem de interesse
    noisy_image = np.where(noise != 128, noise, image)

    # Salva a imagem ruidosa (degradada)
    # IMAGEM A degradada com ruído (IMAGEM B)
    cv2.imwrite('./imagem_a_ruido_salpimenta.png', noisy_image)

    print('Imagem ruidosa salva com sucesso.')

    # Altera o valor nulo da matriz de ruídos para que
    # não haja interferência no histograma
    noise = np.where(noise != 128, noise, -1)

    # Cria o histograma do ruído
    create_histogram('./hist_ruido_salpimenta.png', noise)

    print('Histograma do ruído sal e pimenta salvo.')


def generate_sp_noise(image, salt_prob, pepper_prob):
    """Gera a matriz com ruído do tipo Salt and Pepper.

    Args:
        image: imagem de interesse.

        mu: float que representa a média da função de densidade de
        probabilidade da gaussiana.

        sigma: float que representa o desvio padrão da função de
        densidade de probabilidade da gaussiana.

    Returns:
        matriz: matriz do ruído.
    """

    # Representação unidimensional da matriz de ruído
    # Consideramos 128 como um valor nulo
    # Esse valor indica que não há ruído a ser aplicado
    noise = np.full(shape=image.size, fill_value=128)

    # Atribui uma numeração para cada pixel da matriz
    # de ruídos de referência
    pixel_numbers = np.arange(start=0, stop=image.size)

    # Embaralha o vetor com a numeração dos pixels
    np.random.shuffle(pixel_numbers)

    # Calcula o vetor da função de densidade de probabilidade
    # para cada intensidade de pixel
    probs = sp_pdf(np.arange(256), salt_prob, pepper_prob)

    # Exibe o vetor de p(z)
    print('Resultado da Função p(z)')
    print(probs)

    # Inicia um contador para cada intensidade de pixel
    counter = np.zeros(256)

    # Índice inicial para o vetor com a numeração dos pixels
    curr_pxl = 0

    # Itera sobre todos os valores de intensidade de pixels para
    # uma imagem em escala de cinza
    for curr_int in range(256):

        # Calcula a quantidade de ocorrências da intensidade l
        # na matriz de ruído
        freq = np.ceil(probs[curr_int] * image.size)

        # Enquanto houverem pixels a serem preenchidos e quantidade
        # de pixels com intensidade l for menor que a frequência dela
        while curr_pxl < image.size and counter[curr_int] + 1 <= freq:
            # Atribui a intensidade l no pixel da posição curr_pixel
            # do vetor de pixels numerados a matriz de referência
            noise[pixel_numbers[curr_pxl]] = curr_int

            # Incrementa o contador para intensidade l
            counter[curr_int] += 1

            # Incrementa o índice do vetor dos pixels numerados
            curr_pxl += 1

    # Retorna a matriz do ruído gaussiano com as dimensões da imagem
    return noise.reshape(*image.shape).astype(np.uint8)


def sp_pdf(z, salt_prob, pepper_prob):
    """Função de Densidade de Probabilidade do Ruído Impulsivo.

    Args:
        z: inteiro com um valor aleatório (nível de cinza).
        salt_prob: float a probabilidade de sal (branco).
        pepper_prob: float com a probabilidade de pimenta (preto).

    Returns:
        float: probabilidade/frequência de z.
    """

    # calcula a função de densidade de probabilidade para z
    p = np.zeros_like(z, dtype=np.float32)

    p[0] = pepper_prob
    p[-1] = salt_prob

    # Retorna a probabilidade de z
    return p


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

    # Define uma seed para reprodutibilidade
    np.random.seed(13)

    # Cria uma imagem 5x5 em escala de cinza aleatoriamente
    create_image_a()

    print('Imagem A gerada com sucesso.')

    # Carrega a IMAGEM A
    image = cv2.imread('./imagem_a.png', cv2.IMREAD_GRAYSCALE)

    print('\nRUÍDO GAUSSIANO')

    # Adiciona o ruído a imagem, salva-se a IMAGEM B (ruído)
    # bem como seu histograma
    add_gaussian_noise(image, mu=128., sigma=10.)

    # Carrega a imagem degradada
    noisy_image = cv2.imread(
        './imagem_a_ruido_gaussiano.png', cv2.IMREAD_GRAYSCALE)

    # Cria e salva o histograma da imagem degradada
    create_histogram('./hist_imagem_a_ruido_gaussiano..png', noisy_image)

    print('Histograma da imagem ruidosa salvo.')

    print('\nRUÍDO SAL E PIMENTA')

    apply_sp_noise(image, salt_prob=0.2, pepper_prob=0.2)

    # Carrega a imagem degradada
    noisy_image = cv2.imread(
        './imagem_a_ruido_salpimenta.png', cv2.IMREAD_GRAYSCALE)

    # Cria e salva o histograma da imagem degradada
    create_histogram('./hist_imagem_a_ruido_salpimenta.png', noisy_image)

    print('Histograma da imagem ruidosa salvo.')


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a função principal
    main()
