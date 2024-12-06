import pymupdf as pdf
import matplotlib.pyplot as plt
import numpy as np

def pix_to_image(pix):
    bytes = np.frombuffer(pix.samples, dtype=np.uint8)
    img = bytes.reshape(pix.height, pix.width, pix.n)
    return img

doc = pdf.open("Annotated-hippocampus.pdf") # open a document
for page in doc: # iterate the document pages
    image = page.get_images()[0]
    bbox, matrix = page.get_image_rects(image[0], transform=True)[0]
    bboxpix = page.get_pixmap(clip=bbox)
    im = pix_to_image(bboxpix)
    plt.figure(figsize=(10,10))
    plt.imshow(im)

    shapes = page.get_drawings()
    for shape in shapes:
        for point in shape['items']:
            print(point[1])
            plt.plot(point[1].x, point[1].y, "ro")
    plt.show()