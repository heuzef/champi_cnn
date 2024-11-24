#INIT
from init import *

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

st.set_page_config(page_title="Models", page_icon="🦾")
st.write(r"""<style>.stAppDeployButton { display: none; }</style>""", unsafe_allow_html=True) # Hide Deploy button

models = os.listdir("../../models/artifacts/")
mo_db_path = "../../dataset/"
names_csv_path = "../../notebooks/mushroom_observer/dataset-mushroom-observer/names.csv"

@st.cache_data
def get_champi_name(mo_db_path, names_csv_path):
    """
    Retourne le nom de la classe du champignon depuis le fichier names.csv de Mushroom Observer.
    Requiere numpy, pandas et os.

    Args:
        mo_db_path : Chemin vers le dossier contenant les classes
        names_csv_path : Chemin vers le fichier names.csv

    Returns:
        Dataframe Pandas avec IDs et noms
    """
    # Imports des sources
    data_files = os.listdir(mo_db_path)
    names = pd.read_csv(names_csv_path, delimiter='\t', index_col=0)

    # Recupération des ID des classes
    # champi_classes = []
    # for item in data_files:
    #     champi_classes.append(int(item))

    champi_classes = []
    for item in data_files:
        # Check if the item is a digit (integer)
        if item.isdigit():
            champi_classes.append(int(item))
        else:
            print(f"Skipping non-integer file: {item}")
    
    # Creation du DataFrame
    df = names[["text_name"]].rename(columns={'text_name': 'name'})
    df = df.loc[champi_classes]

    # Resultat
    return df

@st.cache_data
def get_class_names(data_dir):
    dataset = ImageFolder(root=data_dir)
    class_names = dataset.classes
    return class_names

def pred_name(df, classe):
    pred_name = df[df.index == int(classe)]['name'].values[0]
    return pred_name

champi = get_champi_name(mo_db_path, names_csv_path)

def view_model_arch(model):
    # Capture the model summary
    buffer = io.StringIO()  # Create a buffer to hold the summary
    sys.stdout = buffer  # Redirect stdout to the buffer
    model.summary()  # Call model.summary() to populate the buffer
    sys.stdout = sys.__stdout__  # Reset redirect to stdout

    # Get the summary from the buffer
    model_summary = buffer.getvalue()

    with st.expander("Afficher la construction l'architecture CNN", expanded=False):
        st.text(model_summary) # Display the model summary inside the expander

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activation_map = None
        
        # Hook the model to get gradients and activation maps
        self.model.layer4[1].register_backward_hook(self.backward_hook)
        self.model.layer4[1].register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.activation_map = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_index):
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_loss = output[0][class_index]
        class_loss.backward()
        
        # Get the gradients and activation map
        gradients = self.gradients.data.numpy()[0]
        activation = self.activation_map.data.numpy()[0]
        
        # Compute weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Generate CAM
        cam = np.zeros(activation.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activation[i, :, :]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize the CAM
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

def gradcam_keras(model, img_array, pred_index=None, alpha=0.4):
    '''
    Function to visualize Grad-CAM heatmaps and display them over the original image.

    Parameters:
    - model: The trained Keras model.
    - img_array: Preprocessed image array for display.
    - pred_index: Index of the predicted class (optional). If None, the top predicted class is used.
    - alpha: Transparency for heatmap overlay (default is 0.4).
    
    Returns:
    - heatmap: The computed heatmap.
    '''

    # Automatically get the last convolutional layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name or 'conv' in layer.__class__.__name__.lower():
            last_conv_layer_name = layer.name
            break
    
    if last_conv_layer_name is None:
        raise ValueError("No convolutional layer found in the model.")

    # Create a Grad-CAM model
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of the top predicted class for the output feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool the gradients over all the axes leaving only the last one
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weigh the feature map with the pooled gradients
    conv_outputs = conv_outputs[0].numpy()

    # Multiply each channel by the corresponding gradient
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Compute the heatmap by averaging over all channels
    heatmap = np.mean(conv_outputs, axis=-1)

    # ReLU to keep only positive activations and normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Resize heatmap to the original image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.clip(heatmap, 0, 255)

    # Use shape to get the dimensions
    heatmap = Image.fromarray(heatmap).resize((img_array.shape[2], img_array.shape[1]), Image.BILINEAR)
    heatmap = np.array(heatmap)

    # Create a heatmap using Matplotlib
    heatmap = plt.get_cmap('jet')(heatmap / 255.0)[:, :, :3]  # Convert to RGB format
    heatmap = (heatmap * 255).astype(np.uint8)

    # Superimpose the heatmap on the original image
    original_image = np.array(img_array[0])  # Convert from batch dimension
    superimposed_img = heatmap * alpha + original_image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    # Display the result using Streamlit
    st.image(superimposed_img, caption='Grad-CAM', use_column_width=True)

    return heatmap

def gradcam_pytorch(img, model):
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Initialize Grad-CAM
    grad_cam = GradCAM(model)

    # Get the predicted class index
    with torch.no_grad():
        output = model(img_tensor)
    predicted_class_index = output.argmax(dim=1).item()

    # Generate CAM
    cam = grad_cam.generate_cam(img_tensor, predicted_class_index)

    # Ensure CAM is between 0 and 1
    cam = np.clip(cam, 0, 1)
    cam = (cam * 255).astype(np.uint8)  # Convert to 0-255 range for visualization

    # Create a heatmap using the 'jet' colormap
    heatmap = cm.jet(cam)  # Apply the jet colormap
    heatmap = (heatmap[..., :3] * 255).astype(np.uint8)  # Convert to 8-bit RGB

    # Convert the CAM to a PIL image
    cam_image = Image.fromarray(heatmap)  # Create RGB heatmap image

    # Resize the heatmap to match the original image size
    cam_image = cam_image.resize(img.size, Image.LANCZOS)  # Use LANCZOS for high-quality resizing

    # Combine the original image with the heatmap
    overlay = Image.blend(img.convert("RGB"), cam_image, alpha=0.5)  # Overlay the images

    return overlay

# SIDEBAR
st.sidebar.title("Modèle séléctionné")
model_selection = st.sidebar.selectbox("Sélectionnez un modèle :", options=models)
st.sidebar.divider()
st.sidebar.write('Liste des ', len(champi), ' classes du Dataset')
st.sidebar.dataframe(champi, height=850)

# BODY
colored_header(
    label=model_selection.upper(),
    description="Démonstration des modèles entrainés",
    color_name="red-70",
)

# MODELS

if model_selection == 'heuzef_lenet_001.keras':
    # Charger le modèle
    model = tf.keras.models.load_model("../../models/artifacts/"+model_selection)
    class_names = ['1174', '15162', '1540', '2749', '29997', '330', '344', '362', '373', '382', '39842', '42', '50164', '63454']
    
    # Presentation
    st.subheader("Présentation d'un modèle avec une architecture Lenet")
    st.caption("Auteur : [Heuzef](https://heuzef.com) - 2024")
    st.markdown("""
    Nous avons débuté nos expérimentations par un premier modèle avec une architecture LeNet (Y. LeCun et al., 1998) sans attente particulière de performance.
    *(L'entraînement n'ayant pas été effectué sur des données préparées correctement).*

    En effet, nous procédons d'abord par l'augmentation des données puis la division, *(le jeu de validation contient alors des images trop proches de l'entraînement, car simplement modifié par l'augmentation des données).*
    Ainsi, nous obtenons un score d'exactitude de 95%, mais cela ne reflète absolument pas la réalité de la prédiction.

    Ce modèle s'avère donc médiocre et sera rapidement abandonné au profit des algorithmes de transfert learning pour leurs efficacités.
    """)

    view_model_arch(model)

    # Metriques
    st.divider()
    st.subheader("Métriques")
    st.image("../img/lenet_001.png")

    # Prediction
    st.divider()
    st.subheader("Test de prédiction")
    st.warning("""Attention, ce modèle n'est pas entrainé sur toutes les classes du dataset. 
                Les espèces disponible pour ce modèle sont les suivantes : """+str(class_names))

    url = st.text_input("Indiquez l'URL d'une photo pour exécuter le modèle. L'espèce doit appartenir à l'une des classes entrainés.", "https://www.mycodb.fr/photos/Amanita_muscaria_2005_ov_2.jpg")

    if url is not None:

        st.markdown("""
        # 🦾 Exécution !
        """)

        def champi_lenet_predict(url):
            champi_path = tf.keras.utils.get_file(origin=url)
            img = tf.keras.utils.load_img(champi_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            score = predictions[0]
            return int(class_names[np.argmax(score)])

        # Faire une prédiction
        prediction = champi_lenet_predict(url)

        # Afficher la prédiction
        st.info("Résultat de la prédiction : \n\n"+"🔎  ID : "+str(prediction)+" \n\n 🍄  NAME : "+pred_name(champi, prediction).upper())
        st.link_button("🔗 Consulter sur Wikipédia", "https://fr.wikipedia.org/w/index.php?search="+pred_name(champi, prediction))
        st.image(url)

elif model_selection == 'florent_resnet18.pth':

    def load_model(model_path): # Charger le modèle pré-entraîné
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 23)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    # Charger le modèle
    model = load_model('../../models/artifacts/'+model_selection)

    # Presentation
    st.subheader("Présentation du modèle pré-entrainé en Transfert learning avec Resnet18")
    st.caption("Auteur : Florent Constant - 2024")
    st.markdown("""
    Ce modèle à donné un résultat encourageant avec un score d'accuracy sur le jeu de données de validation, suppérieur à 97%.
    Ce dernier à été entrainé en transfert learning depuis le modèle pré-entrainé **ResNet18**, mis en oeuvre a l'aide de Pytorch.

    Outre les bons résultats obtenus, différentes expérimentations ont été réalisées afin d'évaluer la relation entre quantité de données et performance resultante du modèle.
    Une première série d'entrainement a permis d'évaluer l'impact de la data augmentation sur les résultat du modèle :
    - En faisant de l'oversampling pour pour équilibrer le volume de données disponible pour chaque classe (260 images par classe), le score d'accuracy obtenu est de 96.870925684485%.
    - En faisant de l'augmentation pour atteindre un volume de 500 images par classes, le score d'accuracy obtenu est de 97.0013037809648%.
    Ces résultats plutôt contre-intuitifs montrent que das notre cas d'usage la data-augmentation n'a pas permis d'améliorer les résultats du modèle.

    Une seconde serie de tests a été réalisée afin d'évaluer le volume de données nécéssaires pour obtenir des résultats satisfaisant.
    Le volume à été limité à 80, 70, 60, 50, 40, 30, 20 puis 10 images pour chacune des classes. Les résultats se sont montrés surprenant avec :
    - un score équivalent au meilleur score obtenu sur la totalité du dataset, avec seulement 80 images par classes.
    - un score encore au dessus de 90% avec seulement 30 images par classes.
    - un score encore au dessus de 80% avec seulement 10 images par classes.
    """)

    st.image("../img/resnet18_01.png")
    st.image("../img/resnet18_02.png")

    # Metriques
    st.divider()
    st.subheader("Métriques")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Validation - Exactitude", value=0.970)
    col2.metric(label="Validation - Précision", value=0.966)
    col3.metric(label="Validation - F1 score", value=0.965)
    style_metric_cards()

    st.image("../img/conf_matrix_resnet18.png", caption="La matrice de confusion ne fait pas apparaitre de problèmes de confusion entre deux classes particulières")

    # Prediction
    st.divider()
    st.subheader("Test de prédiction")
    uploaded_file = st.file_uploader("Choisissez une photo pour exécuter le modèle. L'espèce doit appartenir à l'une des classes entrainés.", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        st.markdown("""
        # 🦾 Exécution !
        """)

        def preprocess_image(image): # Prétraiter l'image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return transform(image).unsqueeze(0)

        def predict(model, image, class_names): # Faire une prédiction
            image_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            class_index = predicted.item()
            class_name = class_names[class_index]
            return class_name

        image = Image.open(uploaded_file)
        largeur, hauteur = image.size
        if(largeur > hauteur):
            margin = (largeur - hauteur) / 2
            zone_recadrage = (margin, 0, hauteur+margin, hauteur)  
        else:
            margin = (hauteur - largeur) / 2
            zone_recadrage = (0, margin, largeur, largeur+margin)  
        image = image.crop(zone_recadrage)

        # Lire les noms de classes à partir de la structure du répertoire
        class_names = get_class_names(mo_db_path)

        # Faire une prédiction
        prediction = predict(model, image, class_names)

        # Afficher la prédiction
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Image téléchargée")
        with col2:
            imgrgb = Image.open(uploaded_file).convert("RGB")
            heatmap = gradcam_pytorch(imgrgb, model)
            st.image(heatmap, caption='Grad-CAM')

        st.info("Résultat de la prédiction : \n\n"+"🔎  ID : "+str(prediction)+" \n\n 🍄  NAME : "+pred_name(champi, prediction).upper())
        st.link_button("🔗 Consulter sur Wikipédia", "https://fr.wikipedia.org/w/index.php?search="+pred_name(champi, prediction))

elif model_selection == 'heuzef_efficientnetb1_010.keras':
    # Charger le modèle
    model = tf.keras.models.load_model("../../models/artifacts/"+model_selection)
    class_names = ['1174', '15162', '1540', '267', '271', '330', '344', '362', '373', '382', '401', '407', '42', '4920', '53', '939']

    # Presentation
    st.subheader("Présentation du modèle pré-entrainé en Transfert learning avec EfficientNETB1")
    st.caption("Auteur : [Heuzef](https://heuzef.com) - 2024")
    st.markdown("""
    Le transfert learning avec le modèle EfficientNetB1 a permis d'obtenir des résultats satisfaisants malgré les limitations matérielles. 
    En optimisant l'utilisation des ressources, le modèle a pu maintenir, avec un simple CPU AMD Ryzen 7 2700X, de bonnes performances.

    L'utilisation d'un modèle pré-entraîné sur ImageNet fourni une base solide pour la classification. L'entrainement ici est effectué sur **160000 photos, pour 16 classes**.

    Avec optimisation et utilisation de callbacks, le modèle généralise correctement jusqu'à la quatrième epoch avant de subir un overfitting.
    
    **Les performances du modèle ont montré une précision d'entraînement remarquable à 96% et une précision de validation de 86%.**

    Bien que les résultats soient encourageants, ces conclusions ouvrent la voie à des pistes d'amélioration, telles que l'optimisation des hyperparamètres pour affiner les scores de précision sur le jeu d'évaluation ainsi qu'une meilleure gestion des données pour minimiser le risque de sur-apprentissage.
    """)

    with st.expander("Afficher la segmentation du Dataset après ré-échantillonnage"):
        col1, col2 = st.columns(2)
        col1.image("../img/efficientnetb1_dataset.png")
        col2.code("""
            Jeu d'entrainement :
            Found 112000 files

            Jeu de validation :
            Found 48000 files
            
            Jeu de test :
            Found 1360 files

            16 Classes :  
            ['1174', 
            '15162', 
            '1540', 
            '267', 
            '271', 
            '330', 
            '344', 
            '362', 
            '373', 
            '382', 
            '401', 
            '407', 
            '42', 
            '4920', 
            '53', 
            '939']

            batch_size = 32
            """)

    view_model_arch(model)

    # Metriques
    st.divider()
    st.subheader("Métriques")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Test - Exactitude", value=0.928)
    col2.metric(label="Test - Precision", value=0.933)
    col3.metric(label="Test - Recall", value=0.928)
    col4.metric(label="Test - F1-score", value=0.929)
    style_metric_cards()

    st.image("../img/efficientnetb1_metrics.png")
    st.image("../img/efficientnetb1_matrix_02.png")
    st.image("../img/efficientnetb1_predictions.png", caption="Exemples de predictions sur les 16 classes")

    # Prediction
    st.divider()
    st.subheader("Test de prédiction")
    st.warning("""Attention, ce modèle n'est pas entrainé sur toutes les classes du dataset. 
                Les espèces disponible pour ce modèle sont les suivantes : """+str(class_names))
    url = st.text_input("Indiquez l'URL d'une photo pour exécuter le modèle. L'espèce doit appartenir à l'une des classes entrainés.", "https://upload.wikimedia.org/wikipedia/commons/c/cd/Mycena_haematopus_64089.jpg")
    

    if url is not None:
        st.markdown("""
        # 🦾 Exécution !
        """)

        def champi_effnetb1_predict(url):
            champi_path = tf.keras.utils.get_file(origin=url)
            img = tf.keras.utils.load_img(champi_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            score = predictions[0]
            return int(class_names[np.argmax(score)])

        def champi_effnetb1_gradcam(url):
            champi_path = tf.keras.utils.get_file(origin=url)
            img = tf.keras.utils.load_img(champi_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            score = predictions[0]
            heatmap = gradcam_keras(model, img_array, pred_index=np.argmax(score), alpha=0.4)
            return heatmap

        # Faire une prédiction
        prediction = champi_effnetb1_predict(url)

        # Afficher la prédiction
        col1, col2 = st.columns(2)
        with col1:
            st.image(url, caption='Image téléchargée', use_column_width=True)
        with col2:
            champi_effnetb1_gradcam(url)

        st.info("Résultat de la prédiction : \n\n"+"🔎  ID : "+str(prediction)+" \n\n 🍄  NAME : "+pred_name(champi, prediction).upper())
        st.link_button("🔗 Consulter sur Wikipédia", "https://fr.wikipedia.org/w/index.php?search="+pred_name(champi, prediction))

elif model_selection == 'vik_resnet50.h5':
    # Charger le modèle
    model = tf.keras.models.load_model("../../models/artifacts/"+model_selection)
    class_names = ['42', '53', '267', '271','330', '344', '362', '373', '382', '401', '407', '939', '1174', '1540','4920', '15162']  
    index_to_label ={0: '1174', 1: '15162', 2: '1540', 3: '267', 4: '271', 5: '330', 6: '344', 7: '362', 8: '373', 9: '382', 10: '401', 11: '407', 12: '42', 13: '4920', 14: '53', 15: '939'}
    index_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 13: 10, 14: 11, 15: 12, 17: 13, 19: 14, 22: 15}

 
     # Presentation
    st.subheader("Présentation du modèle ResNet50")
    st.caption("Auteur : Viktoriia Saveleva - 2024")
    st.markdown("""
        Le modèle est pré-entraîné sur **ImageNet** (transfer learning). """)
    st.markdown("### Architecture du Modèle")
    with st.expander("Modèle de base", expanded=False):
        st.markdown("""
                      
        1. **Blocs Convolutionnels** : Les couches initiales se composent de couches convolutionnelles qui extraient des caractéristiques de l'image d'entrée en utilisant de petits champs récepteurs.
    
        2. **Blocs Résiduels** : L'innovation majeure de ResNet repose sur l'intégration de blocs résiduels, qui introduisent des connexions de contournement. Ces connexions facilitent le passage des gradients durant l'entraînement, permettant ainsi de former des réseaux de neurones très profonds tout en atténuant le problème de la vanishing gradient.
    
        3. **Normalisation par Lot / Batch Normalization** : Intégrée après chaque couche convolutionnelle, la normalisation par lot normalise la sortie, accélérant l'entraînement et améliorant la convergence.
    
        4. **Fonctions d'Activation** : La fonction d'activation ReLU (Rectified Linear Unit) est appliquée après chaque convolution et normalisation par lot pour introduire de la non-linéarité.
    
        5. **Couches de Pooling** : Les couches de pooling (comme le max pooling) sont utilisées pour réduire les dimensions spatiales des cartes de caractéristiques.

        6. **Couches Entièrement Connectées** : À la fin du réseau, les couches entièrement connectées prennent les décisions de classification finales en fonction des caractéristiques apprises.
                """)
    st.info("**Problème :** surapprentissage\n\n" +
        f"**Précision de validation :** 0.84")

    with st.expander("Modèle personnalisé : Nouvelles Couches", expanded=False):
        st.markdown("""
                         
        7. **Dropouts** : Des couches de dropout (taux de 0.7) sont appliquées après les couches entièrement connectées pour réduire le surapprentissage (overfitting) en désactivant aléatoirement des neurones pendant l'entraînement.

        8. **Régularisation** : La régularisation L2 est appliquée sur les couches denses pour pénaliser les poids trop grands, ce qui aide également à prévenir le surapprentissage.

        9. **Callbacks** : Des rappels (callbacks) sont utilisés pour améliorer le processus d'entraînement :
            - **Early Stopping** : Arrête l'entraînement si la perte de validation ne s'améliore pas pendant un certain nombre d'époques, permettant d'éviter le surapprentissage.
            - **Reduce Learning Rate on Plateau** : Réduit le taux d'apprentissage lorsque la perte de validation atteint un plateau, ce qui permet un ajustement plus fin des poids.
                """)
    st.info("**Problème :** surapprentissage\n\n" +
        f"**Précision de validation :** 0.80")
    with st.expander("Modèle personnalisé : Congélation des Couches", expanded=False):
        st.markdown("""
                     
        10. **Congélation des Couches** : Différentes configurations de congélation des couches ont été testées pour évaluer leur impact sur la performance du modèle :
            - **Congélation Complète** 
            - **Congélation de 5 Couches** : Seules les 5 premières couches sont gelées, permettant aux couches supérieures de s'adapter davantage aux données.
            - **Congélation de 10 Couches** : Une approche intermédiaire, où 10 couches sont gelées.
            - **Congélation de 15 Couches** : Permet un apprentissage plus approfondi en libérant certaines couches supérieures.
            - **Aucune Congélation** : Toutes les couches sont entraînables, offrant la flexibilité maximale pour apprendre des caractéristiques pertinentes. Modèle final.
                """)
    st.info("**Problème :** surapprentissage\n\n" +
        f"**Précision de validation :** 0.84")
    
    with st.expander("Modèle personnalisé : 10 vs 16 classes", expanded=False):
        st.markdown("""
                     
        11. **Augmentation des Données (Classes)** : En augmentant de 10 à 16 classes, le modèle a moins sur-appris. Cela s'explique par une meilleure représentation des données, réduisant ainsi le risque de mémorisation excessive et améliorant sa capacité de généralisation.
                    """)
    st.info("**Problème :** surapprentissage --> est moins prononcé\n\n" +
        f"**Précision de validation :** 0.93")
    
    st.markdown("""
    **Conclusions** : 

    Il n'y a pas d'effet significatif du nombre de couches sur les performances du modèle.
                
    Cependant, augmenter le nombre de classes dans l'entraînement a amélioré l'exactitude du modèle ainsi que d'autres scores.
    """)

    view_model_arch(model)

    # Metriques
    st.divider()
    st.subheader("Métriques")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Test - Accurcay", value=0.9404)
    col2.metric(label="Test - Precision", value=0.9446)
    col3.metric(label="Test - Recall", value=0.9404)
    col4.metric(label="Test - F1-score", value=0.9407)

    style_metric_cards()

    st.image("../img/resnet50_model_last.png")
    st.image("../img/resnet50_cm.png")

    st.markdown("""
    **Conclusions** : 
    D'après la matrice de confusion, certains champignons sont encore mal reconnus.
                
    Prochaines étapes : Entraîner le modèle sur des images des champignons en noir et blanc.
    """)

    # Prediction
    st.divider()
    st.subheader("Test de prédiction")
    st.warning("""Attention, ce modèle n'est pas entrainé sur toutes les classes du dataset. 
                Les espèces disponible pour ce modèle sont les suivantes : """+str(class_names))

    def predict_image_resnet50(uploaded_file):
        # Load and preprocess the image
        img = Image.open(uploaded_file).convert("RGB") # Ensure it's in RGB format
        img = img.resize((224, 224)) # Resize the image
        img_array = tf.keras.preprocessing.image.img_to_array(img) # Convert to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array) # Preprocess for ResNet50

        # Make prediction
        predictions = model.predict(img_array)
    
        # Get predicted class
        predicted_index = np.argmax(predictions, axis=-1)[0] # Get the predicted class index

        predicted_index_2 = index_mapping[predicted_index]
        predicted_class_label = index_to_label[predicted_index_2]

        return img, predicted_class_label, predicted_index_2

    
    # Upload Image
    uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array) # Use ResNet preprocessing
        img, predicted_class_label, predicted_index = predict_image_resnet50(uploaded_file)
        predicted_mushroom_name = pred_name(champi, predicted_class_label)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption='Image téléchargée', use_column_width=True)
        with col2:
            heatmap = gradcam_keras(model, img_array, pred_index=predicted_index, alpha=0.4)

        st.info("Résultat de la prédiction : \n\n" +
                 "🔎  ID : " + str(predicted_class_label) + 
                #f"🔎  Index : {predicted_index} \n\n" + 
                "\n\n 🍄  NAME : " + predicted_mushroom_name.upper())
                #f"\n\n 🍄  NAME : {predicted_class_label.upper()}")
        st.link_button("🔗 Consulter sur Wikipédia", "https://fr.wikipedia.org/w/index.php?search="+predicted_class_label)

elif model_selection == 'yvan_jarvispore.h5':
    # Charger le modèle
    model = tf.keras.models.load_model("../../models/artifacts/"+model_selection)
    # Lire les noms de classes à partir de la structure du répertoire
    class_names = ['01_Agaricus augustus', '02_Amanita augusta', '03_Amanita bisporigera', '04_Amanita muscaria', '05_Amanita velosa', '06_Baorangia bicolor', '07_Bolbitius titubans', '08_Boletinellus merulioides', '09_Boletus edulis', '10_Boletus rex-veris', '11_Cantharellus cinnabarinus', '12_Ceratiomyxa fruticulosa', '13_Craterellus fallax', '14_Flammulina velutipes', '15_Fomitopsis mounceae', '16_Fuligo septica', '17_Ganoderma oregonense', '18_Lactarius indigo', '19_Morchella importuna', '20_Mycena haematopus', '21_Pluteus petasatus', '22_Stropharia ambigua', '23_Trametes versicolor']
    #index_to_label = [0: '01_Agaricus augustus', 1: '02_Amanita augusta', 2: '03_Amanita bisporigera', '04_Amanita muscaria', '05_Amanita velosa', '06_Baorangia bicolor', '07_Bolbitius titubans', '08_Boletinellus merulioides', '09_Boletus edulis', '10_Boletus rex-veris', '11_Cantharellus cinnabarinus', '12_Ceratiomyxa fruticulosa', '13_Craterellus fallax', '14_Flammulina velutipes', '15_Fomitopsis mounceae', '16_Fuligo septica', '17_Ganoderma oregonense', '18_Lactarius indigo', '19_Morchella importuna', '20_Mycena haematopus', '21_Pluteus petasatus', '22_Stropharia ambigua', '23_Trametes versicolor']
    # Presentation
    st.subheader("Présentation du modèle CNN JarviSpore")
    st.caption("Auteur : Yvan Rolland - 2024")
    st.image("../img/jarvispore.png")
    st.write(f"""
    {mention(
        label="Modèle disponible sur HuggingFace", 
        icon="🤗", 
        url="https://huggingface.co/YvanRLD/JarviSpore", 
        write=False
    )}   
    """, unsafe_allow_html=True)

    st.markdown("""
    Suite aux résultats offert par le transfert Learning, nous avons pris l'initiative de créer un modèle de zéro.
    Ce modèle effectue l'entraînement, l'évaluation et l'interprétation d'un modèle de réseau de neurones convolutif (CNN) pour une tâche de classification d'images. 
    
    Les résultats attendues sont :
    * Précision du Modèle : La métrique mesurée est la précision, elle permet de mesurer le pourcentage de classifications correctes effectuées.
    * Interprétabilité avec Grad-CAM : Les heatmaps générées par Grad-CAM doivent indiquer les parties pertinentes de l'image, ce qui aide à comprendre le fonctionnement du modèle.
    * Généralisation : Avec l'utilisation des callbacks et des pondérations de classe, le modèle doit éviter le sur-apprentissage et bien généraliser sur les données de validation et de test.
    """)

    with st.expander("Structure du Code et Méthodologie"):
        st.markdown("""
        Ce projet présente une solution de classification d’images de champignons en utilisant l’apprentissage profond. 
        Le code développé repose sur **TensorFlow** et **Keras** pour créer un réseau de neurones convolutionnel (CNN) 
        optimisé et bien structuré. L’objectif est de classer des images en 23 catégories, correspondant à différentes 
        espèces de champignons. Le modèle est entraîné avec un jeu de données réparti en trois ensembles : **entraînement**, 
        **validation**, et **test**, afin d’assurer la robustesse et la généralisation du modèle.
        
    ### Structure du Code et Méthodologie
    1. **Préparation et Chargement des Données** : Le code exploite `image_dataset_from_directory` de Keras pour charger 
       les images depuis un répertoire, en les redimensionnant et les étiquetant automatiquement. Cette fonction permet 
       également d'appliquer un traitement par lots pour optimiser les performances de l’entraînement. Un soin particulier 
       est apporté à l’équilibrage des classes en calculant les **poids des classes** afin de pallier le déséquilibre naturel 
       des données.
       
    2. **Construction et Optimisation du Modèle CNN** : Le modèle est un **réseau convolutionnel séquentiel** comprenant cinq 
       blocs de convolution avec normalisation par batch et couches de pooling. Ces couches permettent au modèle de capturer 
       efficacement les caractéristiques spatiales de l’image. La régularisation L2 et un dropout de 50% sont appliqués pour 
       éviter le surapprentissage, tandis que la couche finale utilise une activation `softmax` pour la classification multiclasse. 

    3. **Entraînement et Évaluation du Modèle** : Le modèle est compilé avec une fonction de perte `sparse_categorical_crossentropy`, 
       adaptée pour des labels entiers, et un optimiseur Adam. Il est entraîné sur un nombre fixé d’époques avec validation croisée 
       pour surveiller la perte et la précision. Les **callbacks** de Keras permettent de sauvegarder les meilleurs poids et d'arrêter 
       l’entraînement si la performance de validation ne progresse plus, ce qui optimise le temps d'entraînement.

    4. **Visualisation avec Grad-CAM** : Afin d’expliquer les décisions du modèle, la technique de **Grad-CAM** est employée pour 
       générer des heatmaps. Cette approche permet d’identifier les zones de l’image qui influencent le plus les prédictions, 
       apportant une transparence supplémentaire au modèle.

    5. **Interface Interactive pour les Prédictions** : Pour rendre le modèle accessible et interactif, une interface basée sur Tkinter 
       permet de charger une image, de faire une prédiction, et d’afficher la heatmap Grad-CAM. Cette interface offre un aperçu intuitif 
       et visuel des résultats de classification.

    ### Impact et Contribution du Projet
    Ce projet illustre la capacité à déployer une solution de **computer vision** complète, de l’entraînement à l’évaluation et à l’explication 
    des prédictions. Il répond à des enjeux de classification d'images dans un domaine nécessitant une haute précision, avec des ajustements pour 
    équilibrer les classes et une approche pour renforcer l’interprétabilité du modèle. La solution pourrait être étendue à d'autres contextes 
    d'identification d'espèces ou d'analyse d'images médicales, démontrant la portée de ce travail au-delà de la simple reconnaissance d’images.

    """)

        st.code("""
        numpy: 1.26.4
        tensorflow: 2.10.0
        matplotlib: 3.9.2
        scikit-learn: 1.5.2
        PIL: 10.4.0
        cv2: 4.10.0
        pandas: 2.2.3
        python: 3.10.0 (tags/v3.10.0:b494f59, Oct  4 2021, 19:00:18) [MSC v.1929 64 bit (AMD64)]
        cudnn: 64_8
        nvidia_driver: device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:01:00.0, compute capability: 8.6
        gpu_compute_capability: device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:01:00.0, compute capability: 8.6

        """)

    view_model_arch(model)

    # Metriques
    st.divider()
    st.subheader("Métriques")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Entraînement - Exactitude", value=0.977)
    col2.metric(label="Entraînement - Perte", value=0.071)
    col3.metric(label="Validation - Exactitude", value=0.945)
    col4.metric(label="Validation - Perte", value=0.234)
    style_metric_cards()

    st.image("../img/jarvispore_001.png")
    st.image("../img/jarvispore_002.png")
    
    # Prediction
    st.divider()
    st.subheader("Test de prédiction")
    uploaded_file = st.file_uploader("Choisissez une photo pour exécuter le modèle. L'espèce doit appartenir à l'une des classes entrainés.", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.markdown("""
        # 🦾 Exécution !
        """)

        # Fonction pour extraire uniquement le nom de la classe sans l'ID
        def extract_class_name(class_id):
            return class_id.split('_', 1)[1]
        
        def jarvispore_predict(uploaded_file):
            img = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            score = predictions[0]
            return class_names[np.argmax(score)]

        # Faire une prédiction
        prediction = jarvispore_predict(uploaded_file)

        # Extraire uniquement le nom de la classe pour wiki_name
        wiki_name = extract_class_name(prediction)
        
        # Afficher la prédiction
        st.image(uploaded_file, caption='Image téléchargée', use_column_width=True)
        
        st.info("Résultat de la prédiction : \n\n"+"🔎  ID : "+str(prediction))
        st.link_button("🔗 Consulter sur Wikipédia", "https://fr.wikipedia.org/w/index.php?search="+wiki_name)