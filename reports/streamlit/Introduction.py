#INIT
from init import *

st.set_page_config(page_title="CHAMPI CNN", page_icon="🍄", layout="wide")
st.write(r"""<style>.stAppDeployButton { display: none; }</style>""", unsafe_allow_html=True) # Hide Deploy button

# CHAMPI RAIN
rain(
    emoji="🍄",
    font_size=20,
    falling_speed=4,
    animation_length=1,
)

# SIDEBAR
heuzef = mention(
    label="Heuzef",
    icon="🌐",
    url="https://heuzef.com",
    write=False,
)

yvan = mention(
    label="Yvan Rolland",
    icon="github",
    url="https://github.com/YvanRLD",
    write=False,
)

viktoriia = mention(
    label="Viktoriia Saveleva",
    icon="github",
    url="https://github.com/SavelevaV",
    write=False,
)

florent = mention(
    label="Florent Constant",
    icon="github",
    url="https://github.com/FConstantMovework",
    write=False,
)

st.sidebar.title("Crédits")

st.sidebar.write(
    f"""
    {heuzef}
    {yvan}
    {viktoriia}
    {florent}
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    Projet d'étude en datascience accompagné par [DataScientest](https://www.datascientest.com). 
    
    Novembre 2024.
    """)

# BODY
colored_header(
    label="🍄 CHAMPI CNN",
    description="Reconnaissance de champignons à l’aide d'algorithmes de computer vision",
    color_name="red-70",
)

st.markdown(
    """
    ## Sujet

    Les champignons présentant un grand nombre d’espèces et de genre, ainsi une grande variété de caractéristiques biologiques et nutritives, il peut être intéressant de savoir les classifier précisément à l’aide de la computer vision.

    ## Contexte

    La compréhension des champignons est cruciale pour la préservation de la biodiversité, la santé humaine et l'agriculture durable.

    Les champignons ne sont pas des plantes, bien que leur apparente immobilité puisse le faire penser. Une distinction simple est que les champignons ne font pas de photosynthèse, contrairement à une majorité de plantes. 

    En fait, dans l'arbre de la vie, les champignons sont plus proches des animaux que des plantes bien que leur mode de nutrition, paroi cellulaire, reproduction, les distingues également nettement des animaux.

    L'arbre de la vie, qui représente la parenté entre les organismes vivants, peut être découpé en six règnes. Les champignons constituent à eux seuls l'intégralité du **règne FUNGI**, qui rassemblerait hypothétiquement jusqu'à 5 millions d'espèces de champignons. Parmi toutes ces espèces, environ **120000** ont été nommées et “acceptées” par la communauté scientifique [en 2017](https://onlinelibrary.wiley.com/doi/abs/10.1128/9781555819583.ch4).

    La reconnaissance de champignons représente un défi dans le domaine de la vision par ordinateur. En effet, les critères biologiques et le peu d'espèce référencés limite à une reconnaissance peu fiable et sur un échantillon insignifiant si l'on souhaite étudier l'ensemble du règne fongique.
    """)


st.image("https://upload.wikimedia.org/wikipedia/commons/e/e6/Simplified_tree.png?uselang=fr", caption="Arbre phylogénétique. Les champignons constituent à eux seuls l'intégralité du règne FUNGI")