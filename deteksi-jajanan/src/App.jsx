import React, { useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

function App() {
  const imgRef = useRef();
  const [prediction, setPrediction] = useState("");

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const imageURL = URL.createObjectURL(file);
    imgRef.current.src = imageURL;

    const img = new Image();
    img.src = imageURL;
    img.crossOrigin = "anonymous"; // Tambahan jika kamu load gambar dari luar
    img.onload = async () => {
      // Load model
      const model = await tf.loadLayersModel("/model/model.json");

      // Preprocess gambar
      const tensor = tf.browser
        .fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims();

      const predictions = model.predict(tensor);
      const result = await predictions.array();

      const index = result[0].indexOf(Math.max(...result[0]));
      const labels = ["Dadar Gulung", "Kue Lapis", "Risoles"]; // Urutkan sesuai pelatihan
      setPrediction(labels[index]);
    };
  };

  return (
    <div className="App" style={{ textAlign: "center", padding: "20px" }}>
      <h1>Deteksi Jajanan</h1>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      <br />
      <img ref={imgRef} alt="" style={{ maxWidth: "300px", marginTop: "20px" }} />
      <h2>Hasil Prediksi: {prediction}</h2>
    </div>
  );
}

export default App;
