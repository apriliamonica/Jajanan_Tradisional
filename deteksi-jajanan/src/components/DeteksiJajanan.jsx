import React, { useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

/* Label kelas sesuai urutan saat training */
const LABELS = ["Dadar Gulung", "Kue Lapis", "Risoles"];

/* Ukuran input persis seperti di model.json: 150×150 */
const INPUT_SIZE = 150;

export default function DeteksiJajanan() {
  const imgRef = useRef(null);
  const [hasil, setHasil] = useState("-");
  const [model, setModel] = useState(null);
  const [fileName, setFileName] = useState("");

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
    setFileName(file.name);

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
          .expandDims() // -> [1, 150, 150, 3]
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
    <div style={{
      minHeight: '100vh',
      width: '100vw',
      position: 'relative',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'Arial Rounded MT Bold, Arial, sans-serif',
      padding: 0,
      margin: 0
    }}>
      {/* Background image with blur */}
      <div style={{
        position: 'fixed',
        zIndex: 0,
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        background: `url('/makananBG-1.jpg') center center / cover no-repeat`,
        filter: 'blur(7px) brightness(0.85)',
        opacity: 0.85,
        pointerEvents: 'none',
      }} />
      <div style={{
        background: 'rgba(255,255,255,0.97)',
        borderRadius: 20,
        boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
        padding: 36,
        minWidth: 340,
        maxWidth: 370,
        textAlign: 'center',
        margin: 24,
        zIndex: 1,
      }}>
        <h2 style={{
          color: '#d7263d',
          fontWeight: 700,
          letterSpacing: 1,
          marginBottom: 18,
          fontSize: 32
        }}>Jajanan Tradisional</h2>

        <label htmlFor="file-upload" style={{
          display: 'inline-block',
          background: 'linear-gradient(90deg, #ff6347 60%, #ffb347 100%)',
          color: '#fff',
          padding: '12px 28px',
          borderRadius: 10,
          fontWeight: 600,
          fontSize: 18,
          cursor: 'pointer',
          marginBottom: 18,
          boxShadow: '0 2px 8px rgba(255,99,71,0.12)'
        }}>
          Pilih Gambar
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            onChange={handleUpload}
            style={{ display: 'none' }}
          />
        </label>
        {fileName && (
          <div style={{ color: '#888', fontSize: 13, marginBottom: 10 }}>{fileName}</div>
        )}

        {/* Polaroid style preview */}
        {hasil !== '-' && (
          <div style={{
            background: '#fff',
            borderRadius: 14,
            boxShadow: '0 6px 24px 0 rgba(31,38,135,0.10)',
            width: 250,
            margin: '24px auto 0',
            padding: '16px 12px 24px 12px',
            border: '2.5px solid #eee',
            position: 'relative',
            transition: 'box-shadow 0.2s',
          }}>
            <img
              ref={imgRef}
              alt="preview"
              style={{
                width: '100%',
                borderRadius: 10,
                boxShadow: '0 2px 8px #ffb34744',
                background: '#fafafa',
                minHeight: 40,
                display: 'block',
                marginBottom: 14
              }}
            />
            <div style={{
              fontSize: 18,
              color: '#388e3c',
              fontWeight: 700,
              letterSpacing: 0.5,
              textAlign: 'center',
              textShadow: '0 2px 8px #ffb34744',
              marginTop: 0
            }}>{hasil}</div>
          </div>
        )}

        {/* Jika belum ada gambar, tampilkan placeholder polaroid kosong */}
        {hasil === '-' && (
          <div style={{
            background: '#fff',
            borderRadius: 14,
            boxShadow: '0 6px 24px 0 rgba(31,38,135,0.10)',
            width: 250,
            margin: '24px auto 0',
            padding: '16px 12px 24px 12px',
            border: '2.5px solid #eee',
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: 220
          }}>
            <div style={{
              width: '100%',
              height: 140,
              background: 'repeating-linear-gradient(135deg, #ffb34722 0 8px, #fff 8px 16px)',
              borderRadius: 10,
              marginBottom: 14,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ffb347',
              fontSize: 32,
              fontWeight: 700,
              opacity: 0.5
            }}>
              ?
            </div>
            <div style={{
              fontSize: 18,
              color: '#bbb',
              fontWeight: 700,
              letterSpacing: 0.5,
              textAlign: 'center',
              marginTop: 0
            }}>Belum ada gambar</div>
          </div>
        )}
      </div>
      <div style={{marginTop: 12, color: '#fff', fontSize: 13, opacity: 0.8, zIndex: 1}}>
        <span>Jajan &copy; 2025 | Kelompok 1</span>
      </div>
    </div>
  );
}
