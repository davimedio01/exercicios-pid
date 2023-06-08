# Importação dos módulos
import cv2


def main():
    """Função principal do programa.
    """

    # Abre a imagem em escala de cinza
    image = cv2.imread('./img4.png', cv2.IMREAD_GRAYSCALE)

    # Aplica a limiarização de Otsu
    threshold, otsu_image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Exibe o limiar encontrado
    print(f'Limiar utilizado: {threshold}')

    # Salva a imagem limiarizada
    cv2.imwrite('imagem_limiarizada.png', otsu_image)

    # Cria um elemento estruturante retangular 3x3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Aplica a operação morfológica de dilatação
    dilated_image = cv2.morphologyEx(otsu_image, cv2.MORPH_DILATE, kernel)

    # Salva a imagem dilata
    cv2.imwrite('imagem_dilatada.png', dilated_image)

    # Para cada largura W
    for width in range(7, 11):

        # Para cada altura H
        for height in range(7, 11):
            # Cria um elemento estruturante retangular WxH
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))

            # Aplica a operação morfológica de abertura
            new_image = cv2.morphologyEx(dilated_image, cv2.MORPH_OPEN, kernel)

            # Salva a nova imagem
            cv2.imwrite(
                f'nova_imagem_kernel_{width}x{height}.png', new_image)


if __name__ == '__main__':
    """Ponto de entrada do programa.
    """

    # Chama a Função Principal
    main()
