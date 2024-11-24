#INIT
from init import *

st.set_page_config(page_title="EDA", page_icon="📊")
st.write(r"""<style>.stAppDeployButton { display: none; }</style>""", unsafe_allow_html=True) # Hide Deploy button

# SIDEBAR

# BODY
colored_header(
    label="EDA",
    description="Exploration et analyse des données",
    color_name="red-70",
)

st.markdown("""
        ## Source des données
        Après l'identification des différentes sources de données possible, nous concluons que la base de données du site [Mushroom Observer](https://mushroomobserver.org) sera celle qui sera la plus exploitable pour obtenir des données de qualité. 
        
        Le site dispose d'une [API](https://github.com/MushroomObserver/mushroom-observer/blob/main/README_API.md) permettant un accès à la quasi-totalité des données, permettant d'obtenir une visualisation précise du nombre d'espèces répertoriées ainsi que du nombre d'observations et d'images associées à chaque espèce.

        ## Analyse de la taxonomie
        Dans un premier temps, nous avons pris connaissance de la taxonomie des champignons qui se structure en plusieurs niveaux de classification. 
        """)

with st.expander("Afficher un exemple de taxonomie (Champignon de Paris)"):
    st.image("../img/taxon.png")

st.markdown("""
        Nous avons alors conçu un outil d'exploration visuelle qui nous a permis d'évaluer la cohérence existant dans un même ordre, famille ou espèce de champignon, grâce à l'API du service.

        À ce jour, il existe approximativement **35000** genres et de champignon sur terre et certain peuvent compter jusqu'à des milliers d'espèces nommés, tandis que d'autre peuvent n'en compter qu'une seul.
        
        Une analyse visuelle des différents rangs taxonomiques sur des échantillons de photos extraites de Mushroom Observer nous laisse penser que c'est au niveau de l'espèce que nous pouvons observer les plus de traits caractéristiques en vue de réaliser une identification visuelle.
        """)

with st.expander("Visualiser les différences visuelle entres les rangs taxonomique"):
    st.image("../img/visu_diff_01.png", caption="ORDRE des Pezizales")
    st.image("../img/visu_diff_02.png", caption="FAMILLE des Russulaceae")
    st.image("../img/visu_diff_03.png", caption="GENRE Cantharellus")
    st.image("../img/visu_diff_04.png", caption="ESPÈCE Hypholoma lateritium")
    st.divider()
    st.image("../img/visu_diff_05.png", caption="Deux Coprinus comatus, même ESPÈCE, mais visuellement très différents")
    col1, col2  = st.columns(2)
    col1.image("../img/visu_diff_06.jpg")
    col2.image("../img/visu_diff_07.jpg")
    st.caption("""
    Clytocibe nébuleux et Entolome livide sont deux champignons n'appartenant pas au même GENRE
    """)

with st.expander("Visualiser les différences visuelles entre les rangs taxonomiques"):
    st.image("../img/mo_obs_rang.png")

with st.expander("Visualiser les données de Mushroom Observer, groupées par espèce"):
    st.image("../img/mo_img_129_species.png")


# Diagramme circulaire
labels = ['500+ images', '100-500 images', '1-100 images']
values = [130,952,10446]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_layout(
    title_text='Représentation des ESPÈCES en quantité d\'images',
    annotations=[dict(text='Légende', x=1.15, y=1.1, showarrow=False)]
)
st.plotly_chart(fig)

st.markdown("""
    ## Collecte des données
    Parmi les images disponibles, la qualité était très variable avec certaines images où les spécimens étaient mal représentés, en vue microscopique, en schémas, etc ...
    """)

with st.expander("Afficher quelques exemples de photo inexploitables"):
    col1, col2, col3, col4 = st.columns(4)
    col1.image("https://mushroomobserver.org/images/960/1419250.jpg", width=150)
    col2.image("https://images.mushroomobserver.org/960/1106928.jpg", width=150)
    col3.image("https://images.mushroomobserver.org/960/724003.jpg", width=150)
    col4.image("https://images.mushroomobserver.org/960/276963.jpg", width=150)

st.markdown("""
    Suite à ce constat, nous avons donc réalisé une sélection manuelle pour assurer la qualité des images de notre Dataset, en appliquant des critères simples.

    Le spécimen représenté doit :

    - être le sujet principal de l'image.
    - être aisément identifiable par l'œil humain.
    - occuper au moins la moitié de l'image en hauteur et/ou en largeur.
    """)

with st.expander("Afficher quelques exemples de photo avec la qualité recherchée"):
    col1, col2, col3 = st.columns(3)
    col1.image("https://images.mushroomobserver.org/960/262213.jpg", width=200)
    col2.image("https://images.mushroomobserver.org/960/226179.jpg", width=200)
    col3.image("https://images.mushroomobserver.org/960/229897.jpg", width=200)

st.markdown("""
    Ainsi, nous avons développé un outil pour nous aider à réaliser la collecte des données :
    """)

st.video("../img/mo_manual_select_tool.mp4")

col1, col2 = st.columns(2)
col1.link_button(
        "🔬 Accéder à l'outil de constitution des datasets", 
        "http://13.39.133.112:8888/localhost/html/create-qualified-dataset.html"
        )
col2.link_button(
        "🔬 Accéder à l'outil de consultation des datasets", 
        "http://13.39.133.112:8888/localhost/html/show-qualified-dataset.html"
        )
st.markdown("""
    Une fois les photos triées sur le volet, nous avons élaboré un script de webscrapping, exploitant l'API et les restrictions de Mushroom Observer nous permettant d'automatiser la récolte des informations sur les espèces ainsi que le téléchargement de plusieurs milliers de photos HD.

    ## Conclusion

    Grâce à ces outils, ainsi que l'API de Mushroom Observer, nous avons donc :
    * Isoler des espèces de champignons
    * Effectuer une sélection minutieuse sur certaines espèces
    * Réaliser du webscrapping pour télécharger les donnés structuré et non-structurés
    """)