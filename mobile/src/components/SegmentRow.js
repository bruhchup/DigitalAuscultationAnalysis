import React from "react";
import { View, Text, StyleSheet } from "react-native";

const CLASS_COLORS = {
  Normal: "#0891B2",
  Crackle: "#F97316",
  Wheeze: "#EF4444",
  Both: "#8B5CF6",
};

export default function SegmentRow({ segment, index }) {
  const color = CLASS_COLORS[segment.label] || "#64748B";
  const confPct = (segment.confidence * 100).toFixed(0);

  return (
    <View style={styles.row}>
      <Text style={styles.index}>#{index + 1}</Text>
      <Text style={styles.time}>
        {segment.start.toFixed(1)}s - {segment.end.toFixed(1)}s
      </Text>
      <View style={[styles.labelBadge, { backgroundColor: color + "20", borderColor: color }]}>
        <Text style={[styles.labelText, { color }]}>{segment.label}</Text>
      </View>
      <Text style={styles.conf}>{confPct}%</Text>
      {/* Confidence bar */}
      <View style={styles.barBg}>
        <View style={[styles.barFill, { width: `${confPct}%`, backgroundColor: color }]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FFFFFF",
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  index: {
    fontSize: 12,
    color: "#94A3B8",
    width: 28,
    fontWeight: "600",
  },
  time: {
    fontSize: 12,
    color: "#64748B",
    width: 80,
  },
  labelBadge: {
    borderWidth: 1,
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 2,
    marginRight: 8,
  },
  labelText: {
    fontSize: 12,
    fontWeight: "600",
  },
  conf: {
    fontSize: 12,
    color: "#0C4A6E",
    fontWeight: "600",
    width: 36,
    textAlign: "right",
    marginRight: 8,
  },
  barBg: {
    flex: 1,
    height: 6,
    backgroundColor: "#E2E8F0",
    borderRadius: 3,
    overflow: "hidden",
  },
  barFill: {
    height: 6,
    borderRadius: 3,
  },
});
