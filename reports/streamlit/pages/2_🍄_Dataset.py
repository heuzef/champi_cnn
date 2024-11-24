#INIT
from init import *

st.set_page_config(page_title="Dataset", page_icon="🍄")
st.write(r"""<style>.stAppDeployButton { display: none; }</style>""", unsafe_allow_html=True) # Hide Deploy button

data_dir = '../../dataset'
dfNames = pd.read_csv("../../notebooks/mushroom_observer/dataset-mushroom-observer/names.csv", sep="\t")

def get_class_names(data_dir):
    dataset = ImageFolder(root=data_dir)
    class_names = dataset.classes
    return class_names

def openClasGallery(class_name):
        st.session_state.classGallryOpened = class_name

def showAllClasses():
    class_names = get_class_names(data_dir)
    total_img = 0

    container = st.container(border=True)
    col1, col2, col3 ,col4, col5  = container.columns(5)
    cols = [col1, col2, col3 ,col4, col5 ]
    i = 0
    for class_name in class_names:
        class_container = cols[i%5].container(border=True)
        i = i + 1
        image_folder = data_dir + '/' + class_name
        files = os.listdir(image_folder)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        total_img = total_img + len(image_files)

        class_container.write(dfNames[dfNames.id == int(class_name)].text_name.values[0])

        if image_files:
            first_image_file = image_files[0]
            first_image_path = os.path.join(image_folder, first_image_file)
            class_container.image(first_image_path)
            class_container.button("Voir "+str(len(image_files)) + ' images', key= "button_detail_" + class_name, on_click=openClasGallery, args=[class_name])
    
    st.write("Nombre d'images total : ", int(total_img))
    st.write("Nombre de classes : ", len(class_names))
    mean_by_class = round(total_img / len(class_names))
    st.write("Moyenne d'image par classe : ", mean_by_class)

def showClassImages():
    class_name = st.session_state.classGallryOpened
    class_container = st.container(border=True)
    
    classData = {}
    image_folder = data_dir + '/' + class_name
    files = os.listdir(image_folder)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    name = dfNames[dfNames.id == int(class_name)].text_name.values[0] + ' (' + str(len(image_files)) + ' images)'

    class_container.write(name)
    
    class_container.button("Retour", key= "button_detail_" + class_name, on_click=openClasGallery, args=[False])

    col1, col2, col3 ,col4 ,col5 = class_container.columns(5)
    cols = [col1, col2, col3 ,col4 ,col5]
    
    for i in range (0, len(image_files)):
        first_image_file = image_files[i]
        first_image_path = os.path.join(image_folder, first_image_file)
        cols[i%5].image(first_image_path)

def showDataset():
    if 'classGallryOpened' not in st.session_state:
        st.session_state.classGallryOpened = False

    if st.session_state.classGallryOpened == False:
        showAllClasses()
    else:
        showClassImages()

# SIDEBAR

# BODY
colored_header(
    label="Dataset",
    description="Jeu de données final utilisé",
    color_name="red-70",
)

st.markdown("""
    ## Exploration du jeu de données
    """
    )

showDataset()
st.divider()

st.markdown("""
    ## Pré-traitement des données
    Nous développons un second outil pour sélectionner manuellement et précisément un encadrement sur certaines photos :
    """
    )

st.video("../../src/features/manual_picture_bbox_selector/mpbs_demo.mp4")

st.markdown("""
    Le jeu de données est ensuite divisé en 3 jeux de données pour assurer l'entraînement, la validation et le test de nos modèles.

    Finalement, nous réalisons une étape de ré-échantillonnage afin d'augmenter le volume de données d'entraînement et de combler la faible quantité de données en notre possession.
    
    La méthodologie de Data Augmentation utilisé permet d'enrichir les échantillons à partir des images existantes, augmentant ainsi la robustesse et la capacité de généralisation de nos modèles.
    
    La quantité d'images générée varie d'un modèle à l'autre, cependant, nous utilisons globalement des paramètres en communs :
    """
    )

code_DA = """
# Configuration des paramètres pour l'augmentation des images
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),                       # Mise à l'échelle
    layers.RandomFlip("horizontal_and_vertical"),   # Flip
    layers.RandomRotation(0.2),                     # Rotation
    layers.RandomZoom((-0.2, 0.2)),                 # Zoom
    layers.RandomTranslation(0.2, 0.2),             # Translation
    layers.RandomBrightness(factor= [-.001, .001]), # Ajustement de la luminosité
    layers.RandomContrast(factor= .4),              # Ajustement du contraste
])
"""

st.code(code_DA, language="Python")

st.image("../img/images_augmentees.png", caption="Exemple d'images enrichies générées depuis une unique image source, avec la classe ImageDataGenerator de Keras.")
st.divider()
st.image("../img/Layers_structure_rapport2_v2.png", caption="Schématisation de la méthodologie de travail appliquée")