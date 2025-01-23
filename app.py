import streamlit as st
import joblib

# Charger le TF-IDF Vectorizer et le modèle
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Assurez-vous que ce fichier existe dans le même dossier
model = joblib.load('logistic_model.pkl')  # Assurez-vous que ce fichier existe dans le même dossier

# Fonction de nettoyage du texte (assurez-vous de remplacer ceci par la même fonction utilisée dans l'entraînement)
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    return text.strip()

# Configuration de la page Streamlit
st.set_page_config(page_title="Détection de Cyberharcèlement", layout="centered", initial_sidebar_state="collapsed")

# Titre de l'application
st.title("Détection de Cyberharcèlement")
st.write("Cette application utilise un modèle de régression logistique pour détecter les messages potentiellement liés au cyberharcèlement.")

# Saisie utilisateur
message_input = st.text_area("Entrez un message pour l'analyser :", height=150)

if st.button("Analyser"):
    if message_input.strip() == "":
        st.warning("Veuillez entrer un message avant d'analyser.")
    else:
        # Nettoyer et transformer le message
        cleaned_message = clean_text(message_input)
        message_vector = tfidf_vectorizer.transform([cleaned_message])

        # Prédiction
        prediction = model.predict(message_vector)[0]

        # Afficher le résultat
        if prediction == 1:
            st.error("⚠️ Ce message est potentiellement du cyberharcèlement.")
        else:
            st.success("✅ Ce message ne semble pas être du cyberharcèlement.")

# Footer
st.write("---")
st.caption("Projet Mini : Détection de Cyberharcèlement - Créé avec ❤️ par Salima.")
