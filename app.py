import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile

st.set_page_config(page_title="ZampoñAPP", layout="centered")
st.title("🎼 ZampoñAPP: Retroalimentación automática con IA")

st.markdown("Esta herramienta compara tu grabación con un modelo correcto de ejecución de zampoña y te brinda retroalimentación automática sobre afinación y ritmo.")

# Subida de archivos
modelo_file = st.file_uploader("📥 Sube el audio de referencia (modelo)", type=["wav", "mp3"])
estudiante_file = st.file_uploader("📥 Sube tu grabación como estudiante", type=["wav", "mp3"])

if modelo_file and estudiante_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_modelo, tempfile.NamedTemporaryFile(delete=False) as tmp_estudiante:
        tmp_modelo.write(modelo_file.read())
        tmp_estudiante.write(estudiante_file.read())
        modelo_path = tmp_modelo.name
        estudiante_path = tmp_estudiante.name

    # Cargar audios
    modelo, sr = librosa.load(modelo_path, sr=None)
    estudiante, _ = librosa.load(estudiante_path, sr=None)

    # Mostrar duración
    dur_modelo = librosa.get_duration(y=modelo)
    dur_estudiante = librosa.get_duration(y=estudiante)
    diff_duracion = abs(dur_modelo - dur_estudiante)

    # MFCCs para comparar
    mfcc_modelo = librosa.feature.mfcc(y=modelo, sr=sr)
    mfcc_estudiante = librosa.feature.mfcc(y=estudiante, sr=sr)
    min_frames = min(mfcc_modelo.shape[1], mfcc_estudiante.shape[1])
    error_afinacion = np.mean(np.abs(mfcc_modelo[:, :min_frames] - mfcc_estudiante[:, :min_frames]))

    # Resultados
    st.subheader("🔍 Retroalimentación automática")

    if error_afinacion < 20:
        st.success("✅ ¡Muy buena afinación!")
    elif error_afinacion < 50:
        st.warning("⚠️ Afinación aceptable. Puedes mejorar algunas notas.")
    else:
        st.error("❌ La afinación necesita trabajo. Practica con notas largas y lentas.")

    if diff_duracion < 0.5:
        st.success("✅ Ritmo correcto.")
    else:
        st.warning("⚠️ Ritmo irregular. Intenta practicar con un metrónomo.")

    st.markdown(f"📊 **Duración del modelo:** {dur_modelo:.2f}s | **Tu duración:** {dur_estudiante:.2f}s")

    # Gráfico de forma de onda
    st.subheader("📈 Visualización de forma de onda")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(modelo, sr=sr, alpha=0.5, label="Modelo", ax=ax)
    librosa.display.waveshow(estudiante, sr=sr, alpha=0.5, color='r', label="Estudiante", ax=ax)
    ax.set_title("Comparación visual del audio")
    ax.legend()
    st.pyplot(fig)
