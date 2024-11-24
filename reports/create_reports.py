# Chemins des fichiers Markdown à concaténer
file1 = 'champi_rapport_1_eda.md'
file2 = 'champi_rapport_2_modelisation.md'
output_md = 'champi_rapport.md'
output_pdf = 'champi_rapport.pdf'

# Lire et concaténer les fichiers
with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
    content1 = f1.read()
    content2 = f2.read()

# Concaténer les contenus
combined_content = content1 + '\n' + content2

# Écrire dans le nouveau fichier Markdown
with open(output_md, 'w', encoding='utf-8') as f_output:
    f_output.write(combined_content)

print(f"Le nouveau fichier Markdown a été généré : {output_md}")

# Convertir en PDF avec mdpdf
# import os
# os.system(f"mdpdf -o {output_pdf} {output_md}")
# print(f"Le fichier PDF a été généré : {output_pdf}")