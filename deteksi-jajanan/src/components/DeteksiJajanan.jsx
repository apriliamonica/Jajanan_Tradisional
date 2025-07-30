import React, { useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

/* Label kelas sesuai urutan saat training */
const LABELS = ["Dadar Gulung", "Kue Lapis", "Risoles"];

/* Ukuran input persis seperti di model.json: 150×150 */
const INPUT_SIZE = 150;

export default function DeteksiJajanan() {
  const imgRef        = useRef(null);
  const [hasil, setHasil]   = useState("-");
  const [model, setModel]   = useState(null);

  // Muat model 1× saja saat tombol pertama kali diklik
  async function loadModelOnce() {
    if (!model) {
      const loaded = await tf.loadLayersModel("/model/model.json");
      setModel(loaded);
      return loaded;
    }
    return model;
  }

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // tampilkan pratinjau
    const url = URL.createObjectURL(file);
    imgRef.current.src = url;

    const mdl = await loadModelOnce();

    // tunggu gambar selesai di‑render
    imgRef.current.onload = async () => {
      const tensor = tf.tidy(() =>
        tf.browser
          .fromPixels(imgRef.current)
          .resizeNearestNeighbor([INPUT_SIZE, INPUT_SIZE])
          .toFloat()
          .div(255.0)
          .expandDims()               // -> [1, 150, 150, 3]
      );

      const pred = mdl.predict(tensor);
      const data = await pred.data(); // Float32Array panjang 3
      pred.dispose();
      tensor.dispose();

      const maxIdx = data.indexOf(Math.max(...data));
      const persen = (data[maxIdx] * 100).toFixed(2);
      setHasil(`${LABELS[maxIdx]} (${persen} %)`);
    };
  }

  return (
    <div style={{textAlign:"center"}}>
      <h2>🔍 Deteksi Jajanan</h2>

      <input type="file" accept="image/*" onChange={handleUpload} />

      <div style={{marginTop:16}}>
        <img
          ref={imgRef}
          alt=""
          style={{maxWidth:300, borderRadius:8, display: hasil==="-"?"none":"block"}}
        />
      </div>

      <h3 style={{marginTop:12}}>Hasil: {hasil}</h3>
    </div>
  );
}
