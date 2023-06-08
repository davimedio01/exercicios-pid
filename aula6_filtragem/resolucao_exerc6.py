# Importação dos módulos
import cv2
import numpy as np
from scipy import stats
from sklearn import metrics


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
    cv2.imwrite('./imagem_e_ruido_gaussiano.png', noisy_image)


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

    # calcula a função de densidade de probabilidade para z
    p = (1 / (sigma * np.sqrt(2 * np.pi)))
    p *= np.exp(-1 * (((z - mu) ** 2) / (2 * (sigma ** 2))))

    # Retorna a probabilidade de z
    return p


def apply_sp_noise(image, salt_prob, pepper_prob):
    """Aplica Ruído Sal e Pimenta a uma imagem em escala de cinza.

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

    # Adiciona o ruído a imagem de interesse
    noisy_image = np.where(noise != 128, noise, image)

    # Salva a imagem ruidosa (degradada)
    cv2.imwrite('./imagem_e_ruido_salpimenta.png', noisy_image)


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

    # Define a probabilidade de pepper (0)
    p[0] = pepper_prob

    # Define a probabilidade de salt (255)
    p[-1] = salt_prob

    # Retorna a probabilidade de z
    return p


def apply_filters(image, noisy_name, noisy):
    """ Aplica filtros à imagem ruidosa de entrada.

    Args:
        image: matriz da imagem original.
        noisy_name: string com o nome da imagem ruidosa.
        noisy: matriz da imagem ruidosa.
    """

    print('\n+ STATUS DA EXECUÇÃO')

    # Define o tamanho da máscara
    size = 3

    # Aplica o filtro da média
    mean_mask = create_mean_mask(size)
    filtered_image = apply_filter_mask(noisy_name, noisy, 'media', mean_mask)
    print('\n\t* Aplicado o filtro da média.')
    print(
        f'\t\t- MSE original/filtrada = {compute_mse(image, filtered_image)}')

    # Aplica o filtro da mediana
    filtered_image = apply_median_filter(noisy_name, noisy)
    print('\t* Aplicado o filtro da mediana.')
    print(
        f'\t\t- MSE original/filtrada = {compute_mse(image, filtered_image)}')

    # Aplica o filtro da moda
    filtered_image = apply_mode_filter(noisy_name, noisy)
    print('\t* Aplicado o filtro da moda.')
    print(
        f'\t\t- MSE original/filtrada = {compute_mse(image, filtered_image)}')

    # Aplica o filtro gaussiano
    gaussian_mask = create_gaussian_mask(size)
    filtered_image = apply_filter_mask(
        noisy_name, noisy, 'gaussiano', gaussian_mask)
    print('\t* Aplicado o filtro gaussiano.')
    print(
        f'\t\t- MSE original/filtrada = {compute_mse(image, filtered_image)}')

    # Aplica o filtro de máscara h1
    h1_mask = get_h1_mask()
    filtered_image = apply_filter_mask(noisy_name, noisy, 'h1', h1_mask)
    print('\t* Aplicado o filtro passa-alto com máscara h1.')
    print(
        f'\t\t- MSE original/filtrada = {compute_mse(image, filtered_image)}')

    # Aplica o filtro de máscara h2
    h2_mask = get_h2_mask()
    filtered_image = apply_filter_mask(noisy_name, noisy, 'h2', h2_mask)
    print('\t* Aplicado o filtro passa-alto com máscara h2.')
    print(
        f'\t\t- MSE original/filtrada = {compute_mse(image, filtered_image)}')


def apply_filter_mask(image_name, image, filter_name, filter_mask):
    """ Aplica a máscara do filtro à imagem de entrada.

    Args:
        image_name: string com o nome da imagem.
        image: matriz da imagem ruidosa.
        filter_name: string com o nome do filtro.
        filter_mask: matriz com a máscara do filtro.
    """

    # Cria uma cópia da imagem para a aplicação do filtro
    new_image = np.copy(image)

    # Itera em x
    for x in range(1, image.shape[0] - 1):

        # Itera em y
        for y in range(1, image.shape[1] - 1):
            # Aplica a máscara por meio da somatória dos
            # produtos dos pixels com seus respectivos pesos
            sum = image[x - 1][y - 1] * filter_mask[0][0]
            sum += image[x][y - 1] * filter_mask[1][0]
            sum += image[x + 1][y - 1] * filter_mask[2][0]
            sum += image[x - 1][y] * filter_mask[0][1]
            sum += image[x][y] * filter_mask[1][1]
            sum += image[x + 1][y] * filter_mask[2][1]
            sum += image[x - 1][y + 1] * filter_mask[0][2]
            sum += image[x][y + 1] * filter_mask[1][2]
            sum += image[x + 1][y + 1] * filter_mask[2][2]

            # Salva a soma no pixel da nova imagem
            # É importante limitar os valores de 0 a 255
            new_image[x][y] = np.clip(sum, 0, 255)

    # Salva a imagem produzida
    cv2.imwrite(f'{image_name}_{filter_name}.png', new_image)

    # Retorna a imagem obtida
    return new_image


def apply_median_filter(image_name, image):
    """ Aplica o filtro da mediana à imagem de entrada.

    Args:
        image_name: string com o nome da imagem.
        image: matriz da imagem ruidosa.
    """

    # Cria uma cópia da imagem para a aplicação do filtro
    new_image = np.copy(image)

    # Itera em x
    for x in range(1, image.shape[0] - 1):

        # Itera em y
        for y in range(1, image.shape[1] - 1):
            mask = []

            # Adiciona os pixels da vizinhança-8 à máscara
            mask.append(image[x - 1][y - 1])
            mask.append(image[x][y - 1])
            mask.append(image[x + 1][y - 1])
            mask.append(image[x - 1][y])
            mask.append(image[x][y])
            mask.append(image[x + 1][y])
            mask.append(image[x - 1][y + 1])
            mask.append(image[x][y + 1])
            mask.append(image[x + 1][y + 1])

            # Calcula a mediana dos pixels na máscara
            median = np.median(np.array(mask))

            # Salva a mediana no pixel da nova imagem
            new_image[x][y] = median

    # Salva a imagem produzida
    cv2.imwrite(f'{image_name}_mediana.png', new_image)

    # Retorna a imagem obtida
    return new_image


def apply_mode_filter(image_name, image):
    """ Aplica o filtro da moda à imagem de entrada.

    Args:
        image_name: string com o nome da imagem.
        image: matriz da imagem ruidosa.
    """

    # Cria uma cópia da imagem para a aplicação do filtro
    new_image = np.copy(image)

    # Itera em x
    for x in range(1, image.shape[0] - 1):

        # Itera em y
        for y in range(1, image.shape[1] - 1):
            mask = []

            # Adiciona os pixels da vizinhança-8 à máscara
            mask.append(image[x - 1][y - 1])
            mask.append(image[x][y - 1])
            mask.append(image[x + 1][y - 1])
            mask.append(image[x - 1][y])
            mask.append(image[x][y])
            mask.append(image[x + 1][y])
            mask.append(image[x - 1][y + 1])
            mask.append(image[x][y + 1])
            mask.append(image[x + 1][y + 1])

            # Calcula a moda dos pixels na máscara
            mode = stats.mode(mask, keepdims=False)[0]

            # Salva a moda no pixel da nova imagem
            new_image[x][y] = mode

    # Salva a imagem produzida
    cv2.imwrite(f'{image_name}_moda.png', new_image)

    # Retorna a imagem obtida
    return new_image


def create_mean_mask(size):
    """ Cria a máscara do filtro da média (size x size).

    Args:
        size: inteiro indicando o tamanho da máscara.

    Returns:
        matriz: máscara criada.
    """

    return np.ones(shape=(size, size)) / (size ** 2)


def get_h1_mask():
    """ Cria a máscara do filtro h1.

    Returns:
        matriz: máscara criada.
    """

    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])


def get_h2_mask():
    """ Cria a máscara do filtro h2.

    Returns:
        matriz: máscara criada.
    """

    return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])


def create_gaussian_mask(size):
    """ Cria a máscara do filtro gaussiano de dimensão size x size.

    Args:
        size: inteiro que representa a dimensão da máscara.

    Returns:
        matriz: máscara gaussiana gerada.
    """

    # Calcula máscara 1D com base no Binômio de Newton
    nth_row = nth_pascal_triangle_row(size - 1)

    # Divide o vetor pela soma de seus elementos
    nth_row /= np.sum(nth_row)

    # Retorna a máscara 2D através da multiplicação do vetor por sua transposta
    return np.matmul(nth_row.reshape(-1, 1), nth_row.reshape(1, -1))


def nth_pascal_triangle_row(n):
    """ Calcula a n-ésima linha do triângulo de Pascal,
    que contém os coeficientes da Expansão Binomial de grau n.

    Args:
        n: inteiro que indica a linha desejada do triângulo.

    Returns:
        array: n-ésima linha do triângulo.
    """

    # Inicia o vetor com o elemento 1
    nth_row = [1]

    # Itera até N
    for i in range(n):
        # Calcula os valores do Binômio de Newton para cada posição
        nth_row.append(nth_row[i] * (n - i) // (i + 1))

    # Retorna o Binômio de Newton da n-ésima linha
    return np.array(nth_row, dtype=np.float64)


def compute_mse(image1, image2):
    """Calcula o Erro Médio Quadrático (MSE).

    Args:
        image1: imagem original.
        image2: imagem ruidosa.

    Returns:
        float: valor da métrica.
    """

    # Retorna o valor da métrica
    return metrics.mean_squared_error(image1.ravel(), image2.ravel())


def run_gaussian(image):
    print('\n# PARÂMETROS DO RUÍDO GAUSSIANO')

    # Inicializa média e desvio padrão da distribuição gaussiana
    # com valores inválidos
    mu = float(input('\n\t-> Média: '))
    sigma = float(input('\t-> Desvio Padrão: '))

    # Recebe valores de média e de desvio padrão do usuário
    while mu < 0 or mu - 4 * sigma < 0 or mu + 4 * sigma > 255:
        print('\nParâmetros inválidos... Tente novamente.')

        mu = float(input('\n\t-> Média: '))
        sigma = float(input('\t-> Desvio Padrão: '))

    # Adiciona o ruído gaussiano com os valores lidos
    add_gaussian_noise(image, mu, sigma)

    # Carrega a imagem degradada pelo ruído gaussiano
    noisy_g = cv2.imread(
        './imagem_e_ruido_gaussiano.png', cv2.IMREAD_GRAYSCALE)

    # Exibe o Erro Médio Quadrático entre a imagem original e a ruidosa
    print(f'\n\t * MSE original/ruidosa = {compute_mse(image, noisy_g)}')

    # Aplica filtros de máscara 3x3 a essa imagem degradada
    apply_filters(image, 'imagem_e_gaussiano', noisy_g)


def run_salt_pepper(image):
    print('\n# PARÂMETROS DO RUÍDO SAL E PIMENTA')

    # Inicializa as porcentagens de salt e de pepper com
    # com valores inválidos
    salt = float(input('\n\t-> Probabilidade de sal: '))
    pepper = float(input('\t-> Probabilidade de pimenta: '))

    # Recebe as porcentagens de salt e de pepper do usuário
    while salt < 0 or pepper < 0 or salt + pepper > 1:
        print('\nParâmetros inválidos... Tente novamente.')

        salt = float(input('\n\t-> Probabilidade de sal: '))
        pepper = float(input('\t-> Probabilidade de pimenta: '))

    # Adiciona o ruído impulsivo com os valores lidos
    apply_sp_noise(image, salt, pepper)

    # Carrega a imagem degradada pelo ruído impulsivo
    noisy_sp = cv2.imread(
        './imagem_e_ruido_salpimenta.png', cv2.IMREAD_GRAYSCALE)

    # Exibe o Erro Médio Quadrático entre a imagem original e a ruidosa
    print(f'\n\t * MSE original/ruidosa = {compute_mse(image, noisy_sp)}')

    # Aplica filtros de máscara 3x3 a essa imagem degradada
    apply_filters(image, 'imagem_e_salpimenta', noisy_sp)


def main():
    """Função principal do programa
    """

    # Define uma seed para reprodutibilidade
    np.random.seed(13)

    # Carrega imagem E
    image = cv2.imread('./imagem_e.bmp', cv2.IMREAD_GRAYSCALE)

    # Executa a lógica do programa considerando o ruído gaussiano
    run_gaussian(image)

    # Executa a lógica do programa considerando o ruído impulsivo
    run_salt_pepper(image)


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a função principal
    main()

# Entradas: 50 10 // 0.1 0.1
