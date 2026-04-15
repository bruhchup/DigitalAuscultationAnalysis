/**
 * AusculTek API client.
 *
 * Update API_BASE to your server's address:
 *   - Android emulator: http://10.0.2.2:5000
 *   - iOS simulator:    http://localhost:5000
 *   - Physical device:  http://<your-local-ip>:5000
 */

const API_BASE = "http://10.0.2.2:5000"; // change for your setup

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return res.json();
}

export async function classifyAudio(fileUri, fileName) {
  const formData = new FormData();
  formData.append("audio", {
    uri: fileUri,
    name: fileName || "recording.wav",
    type: "audio/wav",
  });

  const res = await fetch(`${API_BASE}/classify`, {
    method: "POST",
    body: formData,
    headers: { "Content-Type": "multipart/form-data" },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(err.error || `Server error: ${res.status}`);
  }

  return res.json();
}
