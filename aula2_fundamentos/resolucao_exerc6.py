import numpy as np
from PIL import Image


def generate_all():
    """Cria todas as imagens (de A a E)
    """

    # Cria a imagem A
    generate_image_a()

    # Cria a imagem B
    generate_image_b()

    # Cria a imagem C
    generate_image_c()

    # Cria a imagem D
    generate_image_d()

    # Cria a imagem E
    generate_image_e()


def generate_image_a():
    """Cria a imagem A com base na função generate_image.

    Returns:
        imagem: imagem composta pela junção dos quadrados.
    """

    # Cria a lista com os níveis de cinza dos quadrados
    intensity = [200]

    # Chama a função base com as especificidades da imagem A
    image = generate_image(1, 1, (256, 256), intensity)

    # Salva a imagem em um arquivo
    image.save('imagem_a.bmp')


def generate_image_b():
    """Cria a imagem B com base na função generate_image.

    Returns:
        imagem: imagem composta pela junção dos retângulos.
    """

    # Chama a função que gera o formato da imagem B
    image = generate_rectangles()

    # Salva a imagem em um arquivo
    image.save('imagem_b.bmp')


def generate_image_c():
    """Cria a imagem C.

    Returns:
        imagem: imagem formada pela modificação do
        formato da imagem B.
    """

    # Chama a função que gera o formato da imagem B e
    # obtém o array da imagem
    image_array = np.asarray(generate_rectangles())

    # Faz uma cópia do array para poder modificá-lo
    image_array = np.copy(image_array)

    # Itera na vertical
    for i in range(53, 74):

        # Itera na horizontal
        for j in range(39, 80):
            # Substitui o nível de cinza para formar os
            # retângulos desejados

            # Retângulo superior esquerdo
            image_array[i][j] = 150

            # Retângulo superior direito
            image_array[i][j + 136] = 150

            # Retângulo inferior esquerdo
            image_array[i + 128][j] = 200

            # Retângulo inferior direito
            image_array[i + 128][j + 136] = 200

    # Obtém o objeto da imagem a partir do array formado
    image = Image.fromarray(image_array.astype(np.uint8), mode='L')

    # Salva a imagem em um arquivo
    image.save('imagem_c.bmp')


def generate_image_d():
    """Cria a imagem D com base na função generate_image.

    Returns:
        imagem: imagem composta pela junção dos quadrados.
    """

    # Cria a lista com os níveis de cinza dos quadrados
    intensity = list(range(150, 226, 25))

    # Chama a função base com as especificidades da imagem D
    image = generate_image(2, 2, (128, 128), intensity)

    # Salva a imagem em um arquivo
    image.save('imagem_d.bmp')


def generate_image_e():
    """Cria a imagem E com base na função generate_image.

    Returns:
        imagem: imagem composta pela junção dos quadrados.
    """

    # Cria a lista com os níveis de cinza dos quadrados
    intensity = list(range(110, 261, 10))
    intensity[-1] -= 5

    # Chama a função base com as especificidades da imagem E
    image = generate_image(4, 4, (64, 64), intensity)

    # Salva a imagem em um arquivo
    image.save('imagem_e.bmp')


def generate_image(per_row, per_column, shape, intensity):
    """Base para a criação das imagens A, B, D e E.

    Args:
        per_row: inteiro que indica o número de retângulos
        presentes em uma linha da imagem final.

        per_column: inteiro que indica o número de retângulos
        presentes em uma coluna da imagem final.

        shape: tupla com dois elementos indicando a altura
        e o comprimento de cada um dos retângulos.

        intensity: lista com os valores de intensidade de
        cada retângulo, sendo que o valor mínimo é 0 e o
        máximo é 255 (imagem em escala de cinza).

    Returns:
        imagem: imagem composta pela junção dos retângulos.
    """

    # Cria um index para o nível de cinza dos retângulos
    idx = 0

    # Cria um array vazio para montar a imagem
    image_array = None

    # Itera as linhas
    for _ in range(per_row):
        # Cria um array vazio para essa nova linha
        new_row = None

        # Itera as colunas
        for _ in range(per_column):
            # Cria um retângulo com as dimensões e a intensidade atual
            new_rect = np.full(shape=shape, fill_value=intensity[idx])

            # Se a linha estiver vazia
            if new_row is None:
                # Inicializa a linha com o retângulo
                new_row = new_rect
            else:
                # Adiciona o retângulo na linha já existente
                new_row = np.hstack([new_row, new_rect])

            # Incrementa o nível de cinza
            idx += 1

        # Se o array da imagem estiver vazio
        if image_array is None:
            # Inicializa a imagem com a linha obtida
            image_array = new_row
        else:
            # Caso contrário, adiciona a linha a imagem já existente
            image_array = np.vstack([image_array, new_row])

    # Cria o objeto do tipo imagem com base no array construído
    return Image.fromarray(image_array.astype(np.uint8), mode='L')


def generate_rectangles():
    """Cria o formato utilizado pelas imagens B e C.

    Returns:
        imagem: imagem composta pela junção de retângulos.
    """

    # Cria a lista com os níveis de cinza dos retângulos
    intensity = list(range(200, 149, -50))

    # Chama a função base com as especificidades dos retângulos
    return generate_image(2, 1, (128, 256), intensity)


def get_image_info(image):
    """Recupera informações sobre a imagem.

    Args:
        image: imagem de interesse.

    Returns:
        tupla: profundidade em bits, densidade de pixels e tamanho.
    """

    # Calcula a profundidade em bits
    bit_depth = compute_bit_depth(image)

    # Calcula a densidade de pixels da imagem no dispositivo
    pixel_density = get_pixel_density(image)

    # Seleciona o tamanho da imagem
    size = image.size

    # Retorna as informações
    return bit_depth, pixel_density, size


def compute_bit_depth(image):
    """Calcula a profundidade em bits da imagem.

    Args:
        image: imagem de interesse.

    Returns:
        inteiro: profundidade em bits.
    """

    # Calcula o número de bandas da imagem
    n_bands = len(image.getbands())

    # Determina o tamanho de um pixel em bytes,
    # sem levar em conta o número de bandas
    pixel_size = np.asarray(image).dtype.itemsize

    # Atualiza o tamanho do pixel considerando todas
    # as bandas presentes na imagem
    pixel_size *= n_bands

    # Para saber a profundidade, baste converter o
    # tamanho obtido para bits
    return pixel_size * 8


def get_pixel_density(image):
    """Calcula a densidade de pixels da imagem no dispositivo.

    Args:
        image: imagem de interesse.

    Returns:
        float: tupla com a densidade de pixels da imagem.
    """

    # Verifica se a imagem tem um dpi definido
    if 'dpi' in image.info:
        # Retorna o dpi
        return image.info['dpi']
    else:
        # Se não, retorna uma tupla que indique a falta de dpi
        return (-1, -1)


def main():
    """Função principal do programa
    """

    # Cria todas as imagens
    generate_all()

    # Cria um dicionário para carregar as imagens
    images_dict = {
        'A': Image.open('./imagem_a.bmp'),
        'B': Image.open('./imagem_b.bmp'),
        'C': Image.open('./imagem_c.bmp'),
        'D': Image.open('./imagem_d.bmp'),
        'E': Image.open('./imagem_e.bmp'),
    }

    # Para evidenciar que o header das imagens geradas com o
    # pacote Pillow, exibe-se a dicionário de informações da imagem
    # As únicas informações presentes são a densidade de pixels (dpi)
    # padrão dos arquivos BMP e o tipo de compressão utilizada (nenhuma)
    print(images_dict['A'].info)
    # Saída: {'dpi': (96.01194815354799, 96.01194815354799), 'compression': 0}

    # Atribui um tamanho real para os objetos físicos representados
    # nas imagens de A a E, considerando que cada imagem contém um
    # tabuleiro/tapete de 100 cm x 100 cm
    object_size = 100

    # Identifica as imagens e exibe suas propriedades
    for image_name, image in images_dict.items():
        # Exibe o nome da imagem
        print(f'\nIMAGEM {image_name}')

        # Obtém a profundidade em bits, a densidade de pixels
        # e o tamanho da imagem
        bit_depth, dpi, size = get_image_info(image)

        # Calcula o profundidade (os níveis de cor)
        color_depth = 2 ** bit_depth

        # Exibe a profundidade em bits e os níveis de cor
        print(f'Profundidade em Bits = {bit_depth}')
        print(f'Níveis de Cor = {color_depth}')

        # Verifica se o dpi da imagem é conhecido
        if dpi[0] != -1:
            print('\nDensidade de Pixels da Imagem no Dispositivo:')

            # Exibe a densidade (horizontal e vertical)
            print(f'-> Horizontal = {dpi[0]:.1f} pixels por polegada')
            print(f'-> Vertical = {dpi[0]:.1f} pixels por polegada')

        print('\nTamanho da Imagem:')

        # Exibe o tamanho (horizontal e vertical)
        print(f'-> Horizontal = {size[0]} pixels')
        print(f'-> Vertical = {size[1]} pixels')

        print('\nResolução Espacial (Taxa de Amostragem):')

        # Exibe a resolução espacial (horizontal e vertical)
        print(f'-> Horizontal = {size[0] / object_size} px/cm')
        print(f'-> Vertical = {size[1] / object_size} px/cm\n')


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a função principal
    main()
