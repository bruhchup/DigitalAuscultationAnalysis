import React from "react";
import { View, Text, StyleSheet } from "react-native";

const CLASS_COLORS = {
  Normal: "#0891B2",
  Crackle: "#F97316",
  Wheeze: "#EF4444",
  Both: "#8B5CF6",
};

export default function ClassBadge({ label, confidence, size = "large" }) {
  const color = CLASS_COLORS[label] || "#64748B";
  const isLarge = size === "large";

  return (
    <View style={[styles.badge, { borderColor: color }, isLarge && styles.badgeLarge]}>
      <Text style={[styles.label, { color }, isLarge && styles.labelLarge]}>
        {label}
      </Text>
      {confidence != null && (
        <Text style={[styles.confidence, isLarge && styles.confidenceLarge]}>
          {(confidence * 100).toFixed(0)}%
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    borderWidth: 2,
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 6,
    alignItems: "center",
    backgroundColor: "#FFFFFF",
  },
  badgeLarge: {
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderRadius: 16,
    borderWidth: 3,
  },
  label: {
    fontSize: 14,
    fontWeight: "700",
  },
  labelLarge: {
    fontSize: 22,
  },
  confidence: {
    fontSize: 12,
    color: "#64748B",
    marginTop: 2,
  },
  confidenceLarge: {
    fontSize: 16,
    marginTop: 4,
  },
});
