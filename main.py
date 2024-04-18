import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create a sidebar with a title and some text
st.sidebar.title('Student Mental Health')
# st.sidebar.write('Choose a page:')

# Add links to different pages
page = st.sidebar.radio('Go to', ['Home', 'Distribusi', 'Hubungan', 'Perbandingan', 'Komposisi', 'Predict'])

# Display different pages based on the user's choice
if page == 'Home':
    st.title('Student Mental Health')
    st.image('data:image/jpeg;base64,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')
    st.write('Dalam bidang kesejahteraan siswa dan keberhasilan akademis,'
             'kumpulan data yang diperoleh dari Kaggle muncul sebagai gudang komprehensif yang menangkap aspek-aspek penting kehidupan siswa. Kumpulan data ini merangkum variabel-variabel penting, termasuk usia, Indeks Prestasi Kumulatif (CGPA), status perkawinan, dan eksplorasi ke dalam dimensi berbeda dari status depresi. Saat kami memulai analisis ini, kumpulan data menjadi sebuah lensa yang dapat digunakan untuk mengungkap interaksi rumit antara faktor pribadi, akademis, dan kesehatan mental yang membentuk pengalaman siswa.')
    st.write('Analisis ini bertujuan untuk melihat pola dan korelasi yang dapat menjadi masukan bagi intervensi yang ditargetkan dan sistem dukungan. Selain itu, memahami dan memberikan pemahaman bagaimana usia, status, nilai akademik, dan status depresi saling terkait memberikan perspektiif holistik, menumbuhkan apresiasi yang lebih dalam terhadap tantangan dan keberhasilan yang mungkin dihadapi siswa selama perjalanan pendidikan mereka.')
    st.write('Situasi bisnis yang mendasari analisis ini adalah kurangnya kesadaran pada kesehatan mental yang berimbas minimnya rasa kepedulian serta apresiasi kepada para siswa yang tengah menghadapi tantangan akademik.')
    st.write('Tujuan dari menavigasi kumpulan data ini adalah untuk menyumbangkan wawasan bermakna yang diiharapkan akan meningkatkan pemahaman kita tentang dinamika siswa namun juga meletakkan dasar untuk mengembangkan strategi yang disesuaikan untuk mendukung kesejahteraan siswa dan keberhasilan akademik.')

elif page == 'Distribusi':
    st.title('Distribusi Page')
    # Load the data
    df = pd.read_csv('StudentMentalhealth.csv')

    # Define the list of columns to visualize
    columns = ['Choose your gender', 'Age', 'Marital status', 'Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']

    # Create the plot
    st.set_option('deprecation.showPyplotGlobalUse', False) # Hide deprecated warning
    plt.figure(figsize=(14, 10))

    for i, quality in enumerate(columns):
        plt.subplot(3, 3, i + 1)
        sns.countplot(data=df, x=quality, palette='viridis')
        plt.xticks(rotation=45)
        plt.tight_layout()

    # Show the plot using Streamlit
    st.pyplot(plt)

    st.write('Dari gambar diatas terlihat bahwa ini adalah halaman distribusi yang berisi grafik batang dan garis yang relevan dengan analisis kesehatan mental.'
             'dengan kolom-kolom data usia, gender, anxiety, depresi, pannick attack, perawatan, status')
    
elif page == 'Hubungan':
    st.title('Hubungan Page')
    # Load the data
    df = pd.read_csv('dataclean.csv')

    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.write('Gambar ini menampilkan sebuah matriks korelasi yang memvisualisasikan hubungan antara berbagai variabel, seperti jenis kelamin, usia, depresi, kecemasan, serangan panik, dan pencarian pengobatan. Matriks ini memberikan wawasan tentang bagaimana faktor-faktor ini saling terkait.')
    
elif page == 'Perbandingan':
    st.title('Perbandingan Page')
    # Load the data
    df = pd.read_csv('dataclean.csv')
    st.write('Gambar ini menampilkan sebuah diagram batang yang menunjukkan “Usia Rata-rata berdasarkan Jenis Kelamin,” dengan dua batang yang memiliki tinggi yang sama, menandakan bahwa usia rata-rata sama untuk kedua jenis kelamin yang diwakili.')

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Choose your gender', y='Age', data=df, estimator=np.mean)
    plt.title('Average Age by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Average Age')
    plt.xticks(rotation=45)
    st.pyplot(plt)

     
elif page == 'Komposisi':
    st.title('Komposisi Page')
    # Load the data
    df = pd.read_csv('dataclean.csv')

    # Select column for composition
    column = 'Age'

    # Count the frequency of each category
    composition = df[column].value_counts()

    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(composition, labels=composition.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Set title
    plt.title(f'Composition of {column}')
    st.pyplot(fig)
    st.write('Gambar diatas menampilkan sebuah diagram pie berjudul “Komposisi Usia” Diagram ini menampilkan distribusi persentase berbagai kelompok usia, yang menarik untuk memvisualisasikan data demografis.'
            'yang memberikan wawasan visual tentang distribusi kelompok usia yang berbeda dalam populasi.')
    
    # -----------
    # Select columns for composition
    columns = ['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']

    # Count the frequency of each category
    composition = df[columns].apply(pd.Series.value_counts)

    # Create a stacked bar chart
    composition.T.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Composition of Mental Health Conditions')
    plt.xlabel('Conditions')
    plt.ylabel('Count')
    plt.legend(title='Status')
    st.pyplot(plt)
    st.write('Diagram ini menunjukkan jumlah orang dengan depresi, kecemasan, dan serangan panik, dibedakan berdasarkan dua status (0 dan 1). Diagram ini relevan untuk memvisualisasikan prevalensi dan koeksistensi kondisi-kondisi ini.')
    
elif page == 'Predict':
    # Load the data
    df = pd.read_csv('dataclean.csv')

    # Gunakan One-Hot Encoding untuk mengubah kolom kategorikal menjadi bentuk numerik
    X = pd.get_dummies(df.drop(['AgeCategory'], axis=1)) 
    y = df[['AgeCategory']]

    # Bagi data menjadi data pelatihan dan data pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi model DecisionTreeClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    
    # Latih model menggunakan data pelatihan
    model.fit(X_train, y_train)

    # Simpan model yang telah dilatih ke dalam file 'knn.pkl'
    with open('knn.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Fungsi untuk memuat model yang telah dilatih
    def load_model():
        with open('knn.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    # Memuat model yang telah dilatih
    model = load_model()

    # Streamlit app
    st.title('Predict Student Mental Health Treatment')
    st.write('Fill in the following information to predict whether a student seeks specialist treatment:')

    gender = st.selectbox('Gender', df['Choose your gender'].unique())
    age = st.number_input('Age', min_value=0, max_value=100, step=1)
    course = st.text_input('Course')
    # year_of_study = st.selectbox('Year of Study', df['Your current year of Study'].unique())
    cgpa = st.number_input('CGPA', min_value=0.0, max_value=4.0, step=0.1)
    # marital_status = st.selectbox('Marital Status', df['Marital status'].unique())
    depression = st.selectbox('Depression', df['Do you have Depression?'].unique())
    anxiety = st.selectbox('Anxiety', df['Do you have Anxiety?'].unique())
    panic_attack = st.selectbox('Panic Attack', df['Do you have Panic attack?'].unique())

    if st.button('Predict'):
        new_data = pd.DataFrame({
            'Choose your gender': [gender],
            'Age': [age],
            'What is your course?': [course],
            # 'Your current year of Study': [year_of_study],
            'What is your CGPA?': [cgpa],
            # 'Marital status': [marital_status],
            'Do you have Depression?': [depression],
            'Do you have Anxiety?': [anxiety],
            'Do you have Panic attack?': [panic_attack]
        })


            # Lakukan One-Hot Encoding untuk kolom kategorikal (jika diperlukan)
        input_data_encoded = pd.get_dummies(new_data)

            # Pastikan kolom-kolom yang dihasilkan dari one-hot encoding pada data pengguna
            # sesuai dengan kolom-kolom yang dihasilkan dari one-hot encoding pada data pelatihan
        missing_columns = set(X_train.columns) - set(input_data_encoded.columns)
        for col in missing_columns:
            input_data_encoded[col] = 0

            # Pastikan urutan kolom-kolom pada data pengguna sesuai dengan urutan kolom-kolom pada data pelatihan
        input_data_encoded = input_data_encoded.reindex(columns=X_train.columns, fill_value=0)

            # Prediksi dengan model yang telah dilatih
        prediction = model.predict(input_data_encoded)

            # Flatten the prediction array to ensure it's 1D
        prediction = prediction.flatten()

            # Tampilkan hasil prediksi sebagai teks
        prediction_text = f"Siswa cenderung akan mencari spesialis untuk pengobatan kesehatan mental.: {prediction[0]}"
        st.write(prediction_text)
        st.write('Yes :1 | No :0')
 