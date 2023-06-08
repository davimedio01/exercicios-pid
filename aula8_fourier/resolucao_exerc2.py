# Importação dos módulos
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import metrics


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

    # Adiciona o ruído a imagem de interesse
    noisy_image = cv2.add(image, noise)

    # Salva a imagem ruidosa (degradada)
    cv2.imwrite('./imagem_ruidosa.png', noisy_image)


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

    # Calcula a função de Densidade de Probabilidade para z
    p = (1 / (sigma * np.sqrt(2 * np.pi)))
    p *= np.exp(-1 * (((z - mu) ** 2) / (2 * (sigma ** 2))))

    # Retorna a probabilidade de z
    return p


def plot_spectrum(dft, filename):
    """Cria uma imagem do espectro da transformada discreta
    de Fourier centrada.

    Args:
        dft: matriz da transformada de Fourier.
        filename: nome do arquivo, sem extensão.

    Returns:
        imagem: imagem do espectro.
    """

    # Calcula as magnitudes
    # 1e-8 evita o cálculo do log 0
    magnitudes = np.log(np.abs(dft) + 1e-8)

    # Configura o gráfico com um determinado intervalo de cores
    plt.imshow(magnitudes, cmap='ocean_r', vmin=6, vmax=12)

    # Cria a barra lateral de cores
    plt.colorbar()

    # Plota e salva o gráfico
    plt.savefig(f'espectro_{filename}.png')

    # Encerra as configurações do gráfico
    plt.close()


def apply_filter1(dft):
    """Aplica o primeiro filtro de suavização proposto.

    Args:
        dft: matriz da transformada de Fourier.
    """

    # Recupera cada posição da transformada de Fourier
    x, y = np.ogrid[:dft.shape[0], :dft.shape[1]]

    # Recupera o ponto central da transformada de Fourier
    center = np.array([int(dft.shape[0] / 2), int(dft.shape[1] / 2)])

    # Calcula a distância euclidiana entre cada posição
    # e o centro da transformada
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Calcula a máscara do filtro
    # Substrai-se de 1 o resultado da divisão das distâncias pela
    # distância máxima (limitando esse resultado de 0 e 1)
    mask = 1 - np.clip(dist / np.max(dist), 0, 1)

    # Aplica a máscara do filtro
    filtered_dft = dft * mask

    # Gera o espectro da nova transformada de Fourier
    plot_spectrum(filtered_dft, 'filtro1')

    # Desfaz o deslocamento da frequência e aplica a transformada inversa,
    # obtendo-se a imagem filtrada
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_dft)))

    # Salva a imagem filtrada
    cv2.imwrite('./imagem_filtrada1.png', filtered_image)


def apply_filter2(dft):
    """Aplica o segundo filtro de suavização proposto.

    Args:
        dft: matriz da transformada de Fourier.
    """

    # Recupera cada posição da transformada de Fourier
    x, y = np.ogrid[:dft.shape[0], :dft.shape[1]]

    # Recupera o ponto central da transformada de Fourier
    center = np.array([int(dft.shape[0] / 2), int(dft.shape[1] / 2)])

    # Calcula a distância euclidiana entre cada posição
    # e o centro da transformada
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Cria a máscara do filtro
    # Inicia-se a máscara com 0 em todas as posições
    # Em seguida, para as posições nas quais a distância é menor do que
    # do que a metade da distância máxima, atualiza o valor para 1
    mask = np.zeros_like(dft)
    mask[dist < np.max(dist) / 2] = 1

    # Aplica a máscara do filtro
    filtered_dft = dft * mask

    # Gera o espectro da nova transformada de Fourier
    plot_spectrum(filtered_dft, 'filtro2')

    # Desfaz o deslocamento da frequência e aplica a transformada inversa,
    # obtendo-se a imagem filtrada
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_dft)))

    # Salva a imagem filtrada
    cv2.imwrite('./imagem_filtrada2.png', filtered_image)


def compute_psnr(image1, image2):
    """Calcula a Relação Sinal-Ruído de Pico (PSNR).

    Args:
        image1: imagem original.
        image2: imagem a ser comparada.

    Returns:
        float: valor da métrica.
    """

    # Retorna o valor da métrica
    return metrics.peak_signal_noise_ratio(image1, image2)


def main():
    """Função principal do programa.
    """

    # Define uma seed para reprodutibilidade
    np.random.seed(13)

    # Realiza a leitura da imagem no formato de matriz
    # e na escala de cinza
    grayscale_image = cv2.imread('./imagem.png', cv2.IMREAD_GRAYSCALE)

    # Salva a imagem convertida para escala de cinza
    cv2.imwrite('./imagem_cinza.png', grayscale_image)

    # Adiciona um ruído gaussiano a imagem
    add_gaussian_noise(grayscale_image, mu=50, sigma=20)

    # Realiza a leitura da imagem ruidosa na escala de cinza
    noisy_image = cv2.imread('./imagem_ruidosa.png', cv2.IMREAD_GRAYSCALE)

    # Calcula a transformada normalizada discreta de Fourier em 2D
    dft = np.fft.fft2(noisy_image)

    # Desloca as frequência para o centro
    dft = np.fft.fftshift(dft)

    # Gera o espectro de Fourier
    plot_spectrum(dft, 'sem_filtro')

    # Aplicar o primeiro filtro proposto
    apply_filter1(dft)

    # Realiza a leitura da imagem filtrada com filtro 1
    filtered_image1 = cv2.imread(
        './imagem_filtrada1.png', cv2.IMREAD_GRAYSCALE)

    # Aplicar o segundo filtro proposto
    apply_filter2(dft)

    # Realiza a leitura da imagem filtrada com filtro 2
    filtered_image2 = cv2.imread(
        './imagem_filtrada2.png', cv2.IMREAD_GRAYSCALE)

    # Calcula as métricas de qualidade entre a imagem original
    # e as demais imagens
    noisy_psnr = compute_psnr(grayscale_image, noisy_image)
    filter1_psnr = compute_psnr(grayscale_image, filtered_image1)
    filter2_psnr = compute_psnr(grayscale_image, filtered_image2)

    # Exibe as métricas de qualidade
    print(f'PSNR entre Imagem Original e Imagem Ruidosa     : {noisy_psnr}')
    print(f'PSNR entre Imagem Original e Imagem com Filtro 1: {filter1_psnr}')
    print(f'PSNR entre Imagem Original e Imagem com Filtro 2: {filter2_psnr}')


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a Função Principal
    main()
