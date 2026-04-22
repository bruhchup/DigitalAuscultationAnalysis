import React, { useState, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
  ScrollView,
} from "react-native";
import { useAudioRecorder, setAudioModeAsync, IOSOutputFormat, AudioQuality } from "expo-audio";
import * as DocumentPicker from "expo-document-picker";
import { classifyAudio } from "../services/api";

export default function HomeScreen({ navigation }) {
  const recorder = useAudioRecorder({
    extension: ".wav",
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 256000,
    android: {
      outputFormat: "default",
      audioEncoder: "default",
    },
    ios: {
      outputFormat: IOSOutputFormat.LINEARPCM,
      audioQuality: AudioQuality.MAX,
      linearPCMBitDepth: 16,
      linearPCMIsBigEndian: false,
      linearPCMIsFloat: false,
    },
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const timerRef = useRef(null);

  // ---- Recording ----

  async function startRecording() {
    try {
      await setAudioModeAsync({
        allowsRecording: true,
        playsInSilentMode: true,
      });

      await recorder.prepareToRecordAsync();
      recorder.record();
      setRecordingDuration(0);

      timerRef.current = setInterval(() => {
        setRecordingDuration((d) => d + 1);
      }, 1000);
    } catch (err) {
      Alert.alert("Error", "Could not start recording: " + err.message);
    }
  }

  async function stopRecording() {
    try {
      clearInterval(timerRef.current);

      await recorder.stop();
      const uri = recorder.uri;

      if (uri) {
        await analyzeFile(uri, "recording.wav");
      }
    } catch (err) {
      Alert.alert("Error", "Could not stop recording: " + err.message);
    }
  }

  // ---- File picker ----

  async function pickFile() {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "audio/*", // Changed to accept any audio type for better compatibility
        copyToCacheDirectory: true,
      });

      if (result.canceled) return;

      const file = result.assets[0];
      // On Android, copyToCacheDirectory usually handles the permission issue,
      // but we ensure we use the uri provided which is now local.
      await analyzeFile(file.uri, file.name);
    } catch (err) {
      Alert.alert("Error", "Could not pick file: " + err.message);
    }
  }

  // ---- Analysis ----

  async function analyzeFile(uri, name) {
    setIsAnalyzing(true);
    try {
      const results = await classifyAudio(uri, name);

      if (results.error) {
        Alert.alert("Analysis Error", results.error);
        return;
      }

      navigation.navigate("Results", { results, filename: name });
    } catch (err) {
      Alert.alert("Connection Error", "Could not reach the server.\n\n" + err.message);
    } finally {
      setIsAnalyzing(false);
    }
  }

  // ---- Render ----

  const formatTime = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Hero */}
      <View style={styles.hero}>
        <Text style={styles.heroTitle}>AusculTek</Text>
        <Text style={styles.heroSubtitle}>Respiratory Sound Analysis</Text>
      </View>

      {/* Recording section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Record Lung Sounds</Text>
        <Text style={styles.sectionDesc}>
          Place a digital stethoscope or device microphone on the patient's chest and record.
        </Text>

        {recorder.isRecording && (
          <View style={styles.timerContainer}>
            <View style={styles.recordingDot} />
            <Text style={styles.timerText}>{formatTime(recordingDuration)}</Text>
          </View>
        )}

        <TouchableOpacity
          style={[styles.recordButton, recorder.isRecording && styles.recordButtonActive]}
          onPress={recorder.isRecording ? stopRecording : startRecording}
          disabled={isAnalyzing}
        >
          <View style={[styles.recordInner, recorder.isRecording && styles.recordInnerActive]} />
        </TouchableOpacity>
        <Text style={styles.recordHint}>
          {recorder.isRecording ? "Tap to stop" : "Tap to record"}
        </Text>
      </View>

      {/* Divider */}
      <View style={styles.divider}>
        <View style={styles.dividerLine} />
        <Text style={styles.dividerText}>or</Text>
        <View style={styles.dividerLine} />
      </View>

      {/* Upload section */}
      <View style={styles.section}>
        <TouchableOpacity
          style={styles.uploadButton}
          onPress={pickFile}
          disabled={isAnalyzing || recorder.isRecording}
        >
          <Text style={styles.uploadIcon}>+</Text>
          <Text style={styles.uploadText}>Upload .wav File</Text>
        </TouchableOpacity>
      </View>

      {/* Analyzing overlay */}
      {isAnalyzing && (
        <View style={styles.analyzing}>
          <ActivityIndicator size="large" color="#0891B2" />
          <Text style={styles.analyzingText}>Analyzing respiratory sounds...</Text>
        </View>
      )}

      {/* History link */}
      <TouchableOpacity
        style={styles.historyButton}
        onPress={() => navigation.navigate("History")}
      >
        <Text style={styles.historyText}>View Analysis History</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F0F9FF",
  },
  content: {
    padding: 24,
    alignItems: "center",
  },
  hero: {
    alignItems: "center",
    marginTop: 20,
    marginBottom: 32,
  },
  heroTitle: {
    fontSize: 32,
    fontWeight: "800",
    color: "#0C4A6E",
  },
  heroSubtitle: {
    fontSize: 14,
    color: "#64748B",
    marginTop: 4,
  },
  section: {
    alignItems: "center",
    width: "100%",
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#0C4A6E",
    marginBottom: 8,
  },
  sectionDesc: {
    fontSize: 13,
    color: "#64748B",
    textAlign: "center",
    marginBottom: 20,
    paddingHorizontal: 20,
  },
  timerContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
  },
  recordingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: "#EF4444",
    marginRight: 8,
  },
  timerText: {
    fontSize: 24,
    fontWeight: "700",
    color: "#0C4A6E",
    fontVariant: ["tabular-nums"],
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 4,
    borderColor: "#CBD5E1",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
  },
  recordButtonActive: {
    borderColor: "#EF4444",
  },
  recordInner: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: "#EF4444",
  },
  recordInnerActive: {
    width: 28,
    height: 28,
    borderRadius: 4,
  },
  recordHint: {
    fontSize: 12,
    color: "#94A3B8",
    marginTop: 8,
    marginBottom: 16,
  },
  divider: {
    flexDirection: "row",
    alignItems: "center",
    width: "100%",
    marginVertical: 20,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: "#CBD5E1",
  },
  dividerText: {
    marginHorizontal: 16,
    color: "#94A3B8",
    fontSize: 13,
  },
  uploadButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
    borderWidth: 2,
    borderColor: "#0891B2",
    borderStyle: "dashed",
    borderRadius: 12,
    paddingVertical: 20,
    paddingHorizontal: 32,
    width: "100%",
  },
  uploadIcon: {
    fontSize: 20,
    color: "#0891B2",
    fontWeight: "700",
    marginRight: 8,
  },
  uploadText: {
    fontSize: 16,
    color: "#0891B2",
    fontWeight: "600",
  },
  analyzing: {
    alignItems: "center",
    marginTop: 32,
  },
  analyzingText: {
    fontSize: 14,
    color: "#0891B2",
    marginTop: 12,
    fontWeight: "500",
  },
  historyButton: {
    marginTop: 32,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    backgroundColor: "#0C4A6E",
  },
  historyText: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 14,
  },
});
