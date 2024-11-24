#INIT
from init import *

st.set_page_config(page_title="HuggingFace", page_icon="🤗")
st.write(r"""<style>.stAppDeployButton { display: none; }</style>""", unsafe_allow_html=True) # Hide Deploy button

# SIDEBAR
st.sidebar.title("Modèle utilisé")
st.sidebar.markdown("""
    Mushrooms image detection ViT par dima806.

    Partagé sous licence apache-2.0 le 15/04/2024.
    """)

st.sidebar.write(
    f"""
    {mention(
        label="Détail du modèle sur HuggingFace", 
        icon="🤗", 
        url="https://huggingface.co/dima806/mushrooms_image_detection", 
        write=False
    )}
    {mention(
        label="Détail du Dataset utilisé sur Kaggle", 
        icon="🗃️", 
        url="https://www.kaggle.com/datasets/thehir0/mushroom-species", 
        write=False
    )}    
    """,
    unsafe_allow_html=True,
)

# BODY
colored_header(
    label="HuggingFace",
    description="Utilisation d'un modèle communautaire",
    color_name="red-70",
)

st.markdown("""
    Nous expérimentons l'utilisation d'un modèle communautaire de classification pré-entraîné sur **100 espèces** de champignons russe.

    Ce modèle, partagé par [Dmytro Iakubovskyi](https://huggingface.co/dima806), est entraîné sur **233480** images et se base sur l'architecture **Vision Transformer (ViT)** *(85.9M de paramètres)*.
    """)

# Importer le modèle
processor = AutoImageProcessor.from_pretrained("dima806/mushrooms_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/mushrooms_image_detection")

with st.expander("Afficher l'architecture du modèle"):
    st.code(model)

st.subheader("Rapport de classification")

def metrics():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Exactitude", value=0.899)
    col2.metric(label="Precision", value=0.905)
    col3.metric(label="Recall", value=0.899)
    col4.metric(label="F1-score", value=0.896)
    style_metric_cards()

metrics()

with st.expander("Afficher la matrice de confusion"):
    st.image("https://cdn-uploads.huggingface.co/production/uploads/6449300e3adf50d864095b90/_tfRCaKBzs3rx82PT2xX2.png", caption="Basé sur plus de 50 000 photos prises en Russie")


st.subheader("Liste des 100 espèces utilisées pour l'entraînement")

dict_labels = {'Urnula craterium': 0, 'Leccinum albostipitatum': 1, 'Lactarius deliciosus': 2, 'Clitocybe nebularis': 3, 'Hypholoma fasciculare': 4, 'Lactarius torminosus': 5, 'Lycoperdon perlatum': 6, 'Verpa bohemica': 7, 'Schizophyllum commune': 8, 'Leccinum aurantiacum': 9, 'Phellinus igniarius': 10, 'Suillus luteus': 11, 'Coltricia perennis': 12, 'Cetraria islandica': 13, 'Amanita muscaria': 14, 'Pholiota aurivella': 15, 'Trichaptum biforme': 16, 'Artomyces pyxidatus': 17, 'Calocera viscosa': 18, 'Sarcosoma globosum': 19, 'Evernia prunastri': 20, 'Laetiporus sulphureus': 21, 'Lobaria pulmonaria': 22, 'Bjerkandera adusta': 23, 'Vulpicida pinastri': 24, 'Imleria badia': 25, 'Evernia mesomorpha': 26, 'Physcia adscendens': 27, 'Coprinellus micaceus': 28, 'Armillaria borealis': 29, 'Trametes ochracea': 30, 'Cantharellus cibarius': 31, 'Pseudevernia furfuracea': 32, 'Tremella mesenterica': 33, 'Gyromitra infula': 34, 'Leccinum versipelle': 35, 'Mutinus ravenelii': 36, 'Pholiota squarrosa': 37, 'Amanita rubescens': 38, 'Amanita pantherina': 39, 'Sarcoscypha austriaca': 40, 'Boletus edulis': 41, 'Coprinus comatus': 42, 'Merulius tremellosus': 43, 'Stropharia aeruginosa': 44, 'Cladonia fimbriata': 45, 'Suillus grevillei': 46, 'Apioperdon pyriforme': 47, 'Cerioporus squamosus': 48, 'Leccinum scabrum': 49, 'Rhytisma acerinum': 50, 'Hypholoma lateritium': 51, 'Flammulina velutipes': 52, 'Tricholomopsis rutilans': 53, 'Coprinopsis atramentaria': 54, 'Trametes versicolor': 55, 'Graphis scripta': 56, 'Ganoderma applanatum': 57, 'Phellinus tremulae': 58, 'Peltigera aphthosa': 59, 'Parmelia sulcata': 60, 'Fomitopsis betulina': 61, 'Pleurotus pulmonarius': 62, 'Fomitopsis pinicola': 63, 'Daedaleopsis confragosa': 64, 'Hericium coralloides': 65, 'Trametes hirsuta': 66, 'Coprinellus disseminatus': 67, 'Kuehneromyces mutabilis': 68, 'Pleurotus ostreatus': 69, 'Phlebia radiata': 70, 'Boletus reticulatus': 71, 'Phallus impudicus': 72, 'Macrolepiota procera': 73, 'Fomes fomentarius': 74, 'Suillus granulatus': 75, 'Gyromitra esculenta': 76, 'Xanthoria parietina': 77, 'Nectria cinnabarina': 78, 'Sarcomyxa serotina': 79, 'Inonotus obliquus': 80, 'Panellus stipticus': 81, 'Hypogymnia physodes': 82, 'Hygrophoropsis aurantiaca': 83, 'Cladonia rangiferina': 84, 'Platismatia glauca': 85, 'Calycina citrina': 86, 'Cladonia stellaris': 87, 'Amanita citrina': 88, 'Lepista nuda': 89, 'Gyromitra gigas': 90, 'Crucibulum laeve': 91, 'Daedaleopsis tricolor': 92, 'Stereum hirsutum': 93, 'Paxillus involutus': 94, 'Lactarius turpis': 95, 'Chlorociboria aeruginascens': 96, 'Chondrostereum purpureum': 97, 'Phaeophyscia orbicularis': 98, 'Peltigera praetextata': 99}

df_labels = pd.DataFrame.from_dict(dict_labels, orient='index', columns=['ID'])

col1, col2 = st.columns(2)
col1.dataframe(df_labels, use_container_width=False)
col2.image("https://storage.googleapis.com/kaggle-datasets-images/3708144/6427080/8e3f306869ae228d793494929eb27499/dataset-cover.png", caption="Basé sur plus de 50 000 photos prises en Russie")

# Prediction
st.divider()
st.subheader("Test de prédiction")
uploaded_file = st.file_uploader("Choisissez une photo pour exécuter le modèle. L'espèce doit appartenir à l'une des classes entraînés.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("""
    # 🦾 Exécution !
    """)

    def champi_predict_dima806_mid(img):
        image = PIL.Image.open(img)
        inputs = processor(image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_index = logits.argmax(-1).item()
        return predicted_class_index
    
    def champi_label(dict_labels, class_index):
        for cle, valeur in dict_labels.items():
            if valeur == class_index:
                return cle
        return None

    prediction = champi_predict_dima806_mid(uploaded_file)

    # Afficher la prédiction
    st.info("Résultat de la prédiction : \n\n"+"🔎  ID : "+str(prediction)+" \n\n 🍄  NAME : "+champi_label(dict_labels, prediction).upper())
    st.link_button("🔗 Consulter sur Wikipédia", "https://fr.wikipedia.org/w/index.php?search="+champi_label(dict_labels, prediction))
    st.image(uploaded_file)