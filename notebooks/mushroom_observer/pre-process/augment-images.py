from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

if not os.path.isdir("data/augmented"):
    os.makedirs("data/augmented")

# Chargez l'image originale
img = load_img('data/cropped/362/212919.jpg')

# Convertissez l'image en tableau
x = img_to_array(img)

# Vérifiez la forme du tableau
print(x.shape)

# Créez une instance de ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=.15,
        height_shift_range=.15,
        shear_range=35,
        zoom_range=.3,
        horizontal_flip=True,
        fill_mode='reflect')

# Générez un ensemble d'images augmentées à partir de l'image originale
datagen.fit(x.reshape((1,) + x.shape))
it_gen = datagen.flow(x.reshape((1,) + x.shape), batch_size=1, save_to_dir='data/augmented', save_prefix='img_flow', save_format='jpeg')


# Itérez sur le générateur et enregistrez chaque image sur le disque
for i in range(60):
    img_gen = next(it_gen)
    img = array_to_img(img_gen[0], scale=True)
    #img.save('data/augmented/img_{}.jpg'.format(i))
