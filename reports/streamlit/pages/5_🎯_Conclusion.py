#INIT
from init import *

st.set_page_config(page_title="Conclusion", page_icon="🎯")
st.write(r"""<style>.stAppDeployButton { display: none; }</style>""", unsafe_allow_html=True) # Hide Deploy button

# SIDEBAR
st.sidebar.title("Librairies utilisés")

with open('../../requirements.txt', 'r') as fichier:
    st.sidebar.code(fichier.read(), language="python")

st.sidebar.divider()

st.sidebar.link_button("📖 Rapport détaillé du projet (PDF)", "https://git.heuzef.com/heuzef/jan24_cds_mushrooms/src/branch/master/reports/champi_cnn_report.pdf", type='primary')
st.sidebar.link_button("👩‍💻 Dépôt GIT du projet", "https://git.heuzef.com/heuzef/jan24_cds_mushrooms", type='primary')

# BODY
colored_header(
    label="Conclusion",
    description="Interprétation des résultats",
    color_name="red-70",
)

st.image("../img/champicnn.jpg")

st.markdown("""
    ### Comparaison des résultats

    Les différents essais réalisés ont mis en évidence d'importantes différences de résultats obtenus avec divers modèles sur un même jeu de données. 
    
    Alors que certains modèles, affichent des limites significatives pour ce cas d'utilisation, d'autres, ont démontré d'excellentes performances. 
    
    En poursuivant notre analyse, nos comparaisons soulignent qu'un modèle plus profond n'est pas toujours synonyme de meilleures performances ; au contraire, sur un petit jeu de données, un modèle plus complexe peut s'avérer moins performant.

    Nous pouvons noter que si les modèles avec une architecture les plus basiques (Lenet) offrent des résultats très moyens, certains modèles entraînables en transfert learning offrent cependant de bonnes performances. L'enjeu est alors de sélectionner le bon modèle, tous ne sont pas adaptés. Certains comme VGG16 se sont montrés rapidement sujet à un sur-apprentissage, avec de faibles performances sur le jeu de validation, malgré de nombreux essais avec différentes techniques d'optimisation.

    Nous notons aussi que la conception d'un modèle sur mesure, avec une architecture complexe, bien que très fastidieux, permet également d'obtenir des métriques et rapports de classification performant à tous les niveaux.

    Nous concluons donc que dans notre cas le transfert learning ou la création d'un modèle sur mesure nous ont permis d'obtenir des résultats intéressants, mais que le transfert learning et plus simple, rapide et moins onéreux à mettre en œuvre.
    """)

col1, col2 = st.columns(2)
col1.image("../img/mlflow.png", caption="Serveur de tracking MLFlow, pour le suivi et la gestion et la centralisation de nos expériences")
col2.image("../img/mlflow2.png", caption="Expérimentations sur le modèle VGG16")

st.link_button("🔗 Accéder au serveur de tracking MLFlow", "https://champi.heuzef.com", type='primary')

st.divider()

st.markdown("""
    ### Interprétabilité

    Pour mieux comprendre les résultats et les décisions prises par les algorithmes, nous avons utilisé Grad-CAM (Gradient-weighted Class Activation Mapping), une technique puissante d'interprétation des modèles de deep learning, en particulier pour la classification d'images. Cette méthode permet de visualiser les régions d'une image qui influencent le plus les décisions d'un modèle. 
    
    En générant des cartes thermiques (heatmaps) superposées sur les images d'entrée, Grad-CAM met en évidence les caractéristiques jugées essentielles par le modèle pour ses prédictions.

    Les différentes évaluations des modèles effectués sur des données de tests montrent que de façon globale, les modèles ont tous tendance à effectuer de la sur-interprétation et gèrent particulièrement mal les couleurs des champignons. 
    
    En effet, les méthodes Grad-Cam permettent de visualiser cette tendance à prendre en compte des zones précises, sans se concentrer sur les zones très colorées. 
    
    La couleur est pourtant l'un des points les plus importants, les modèles montrent cependant tous de grandes faiblesse ici. Les performances mesurées durant la reproduction des expériences avec des photos monochrome appuient cette hypothèse.
    """)

st.image("../img/gradcam.png", caption="Ces résultats soulignent l'importance d'une analyse approfondie pour mieux comprendre les performances de chaque modèle dans des situations variées")

st.image("../img/jarvispore_004.png", caption="Ici, la couleur bleue, propre au Lactarius indigo, n'est pas prise en compte")

st.image("../img/champi_diff.png", caption="Ces deux espèces se distinguent essentiellement par la nuance de couleur, amènent à des erreurs de prédiction")