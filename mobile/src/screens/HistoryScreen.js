import React, { useState } from "react";
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from "react-native";

const CLASS_COLORS = {
  Normal: "#0891B2",
  Crackle: "#F97316",
  Wheeze: "#EF4444",
  Both: "#8B5CF6",
};

/**
 * In-memory history. In a production app this would use AsyncStorage or SQLite.
 * The App component passes history/setHistory as params via navigation context.
 */
export default function HistoryScreen({ route, navigation }) {
  const history = route.params?.history || [];
  const setHistory = route.params?.setHistory;

  function clearHistory() {
    Alert.alert("Clear History", "Remove all past analyses?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Clear",
        style: "destructive",
        onPress: () => setHistory && setHistory([]),
      },
    ]);
  }

  function viewResult(item) {
    navigation.navigate("Results", {
      results: item.results,
      filename: item.filename,
    });
  }

  function renderItem({ item, index }) {
    const color = CLASS_COLORS[item.results.overall_label] || "#64748B";

    return (
      <TouchableOpacity style={styles.card} onPress={() => viewResult(item)}>
        <View style={[styles.indicator, { backgroundColor: color }]} />
        <View style={styles.cardBody}>
          <Text style={styles.cardFilename} numberOfLines={1}>
            {item.filename}
          </Text>
          <View style={styles.cardRow}>
            <Text style={[styles.cardLabel, { color }]}>{item.results.overall_label}</Text>
            <Text style={styles.cardConf}>
              {(item.results.overall_confidence * 100).toFixed(0)}%
            </Text>
            <Text style={styles.cardMeta}>
              {item.results.total_cycles} segs | {item.results.duration_sec}s
            </Text>
          </View>
          <Text style={styles.cardTime}>{item.time}</Text>
        </View>
      </TouchableOpacity>
    );
  }

  return (
    <View style={styles.container}>
      {history.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyTitle}>No analyses yet</Text>
          <Text style={styles.emptyDesc}>
            Record or upload a lung sound from the home screen to get started.
          </Text>
          <TouchableOpacity
            style={styles.goHomeButton}
            onPress={() => navigation.navigate("Home")}
          >
            <Text style={styles.goHomeText}>Go to Home</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          <FlatList
            data={[...history].reverse()}
            keyExtractor={(_, i) => String(i)}
            renderItem={renderItem}
            contentContainerStyle={styles.list}
          />
          <TouchableOpacity style={styles.clearButton} onPress={clearHistory}>
            <Text style={styles.clearText}>Clear History</Text>
          </TouchableOpacity>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F0F9FF",
  },
  list: {
    padding: 16,
  },
  card: {
    flexDirection: "row",
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    marginBottom: 10,
    overflow: "hidden",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 3,
    elevation: 2,
  },
  indicator: {
    width: 5,
  },
  cardBody: {
    flex: 1,
    padding: 14,
  },
  cardFilename: {
    fontSize: 14,
    fontWeight: "700",
    color: "#0C4A6E",
    marginBottom: 4,
  },
  cardRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  cardLabel: {
    fontWeight: "700",
    fontSize: 13,
  },
  cardConf: {
    fontSize: 13,
    color: "#64748B",
  },
  cardMeta: {
    fontSize: 12,
    color: "#94A3B8",
  },
  cardTime: {
    fontSize: 11,
    color: "#94A3B8",
    marginTop: 4,
  },
  empty: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 40,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#0C4A6E",
    marginBottom: 8,
  },
  emptyDesc: {
    fontSize: 14,
    color: "#64748B",
    textAlign: "center",
    marginBottom: 24,
  },
  goHomeButton: {
    backgroundColor: "#0C4A6E",
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 28,
  },
  goHomeText: {
    color: "#FFFFFF",
    fontWeight: "600",
  },
  clearButton: {
    alignSelf: "center",
    marginBottom: 24,
    paddingVertical: 10,
    paddingHorizontal: 20,
  },
  clearText: {
    color: "#EF4444",
    fontWeight: "600",
    fontSize: 14,
  },
});
