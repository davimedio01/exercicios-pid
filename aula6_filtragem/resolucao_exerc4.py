# Importação dos módulos
import numpy as np


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

    # Calcula o desvio padrão
    stdev = np.sqrt((size - 1) / 2)

    # Retorna a máscara 2D através da multiplicação do vetor por sua transposta
    return np.matmul(nth_row.reshape(-1, 1), nth_row.reshape(1, -1)), stdev


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


def main():
    """Função principal do programa
    """

    # Cria um array de tuplas para armazenar as saídas
    outputs = []

    # Gera as máscaras dos filtros gaussianos e calcula seus desvios padrões
    outputs.append(('3x3', *create_gaussian_mask(size=3)))
    outputs.append(('5x5', *create_gaussian_mask(size=5)))
    outputs.append(('7x7', *create_gaussian_mask(size=7)))

    # Itera o array de saídas
    for dim, mask, stdev in outputs:
        # Exibe a dimensão da máscara
        print(f'\nMáscara gaussiana {dim}:')

        # Exibe a matriz da máscara
        print(mask)

        # Exibe o desvio padrão da máscara
        print(f'Desvio padrão = {stdev}')

        # Exibe o separador
        print('----------')


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a função principal
    main()
