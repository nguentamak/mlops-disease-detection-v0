# Utilise une image Python légère
FROM python:3.10.0rc2

# Définit le répertoire de travail
WORKDIR /app

# Copie les fichiers nécessaires
COPY requirements.txt .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste des fichiers
COPY . .

# Verifie que les fichiers ont bien été copiés
RUN ls -R /app

# Lance l'application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
