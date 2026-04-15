import React from "react";
import { View, Text, ScrollView, StyleSheet, TouchableOpacity } from "react-native";
import ClassBadge from "../components/ClassBadge";
import MetricCard from "../components/MetricCard";
import SegmentRow from "../components/SegmentRow";

const CLASS_COLORS = {
  Normal: "#0891B2",
  Crackle: "#F97316",
  Wheeze: "#EF4444",
  Both: "#8B5CF6",
};

export default function ResultsScreen({ route, navigation }) {
  const { results, filename } = route.params;
  const isNormal = results.overall_label === "Normal";

  // Count per class for summary
  const classCounts = {};
  for (const c of results.cycles) {
    classCounts[c.label] = (classCounts[c.label] || 0) + 1;
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <Text style={styles.filename}>{filename}</Text>

      {/* Overall result */}
      <View style={styles.overallCard}>
        <ClassBadge label={results.overall_label} confidence={results.overall_confidence} size="large" />

        {/* Status banner */}
        <View style={[styles.statusBanner, isNormal ? styles.statusNormal : styles.statusAbnormal]}>
          <Text style={[styles.statusText, isNormal ? styles.statusTextNormal : styles.statusTextAbnormal]}>
            {isNormal
              ? `No abnormalities detected across ${results.total_cycles} segments.`
              : `${results.abnormal_cycles} of ${results.total_cycles} segments (${Math.round(
                  (results.abnormal_cycles / results.total_cycles) * 100
                )}%) show abnormalities.`}
          </Text>
        </View>
      </View>

      {/* Metrics row */}
      <View style={styles.metricsRow}>
        <MetricCard title="Segments" value={results.total_cycles} />
        <MetricCard title="Duration" value={`${results.duration_sec}s`} />
        <MetricCard title="Normal" value={results.normal_cycles} />
        <MetricCard title="Abnormal" value={results.abnormal_cycles} />
      </View>

      {/* Class distribution */}
      <Text style={styles.sectionTitle}>Classification Breakdown</Text>
      <View style={styles.distributionRow}>
        {Object.entries(classCounts).map(([label, count]) => (
          <View key={label} style={styles.distItem}>
            <View style={[styles.distDot, { backgroundColor: CLASS_COLORS[label] || "#64748B" }]} />
            <Text style={styles.distLabel}>{label}</Text>
            <Text style={styles.distCount}>{count}</Text>
          </View>
        ))}
      </View>

      {/* Segment details */}
      <Text style={styles.sectionTitle}>Segment Details</Text>
      {results.cycles.map((seg, i) => (
        <SegmentRow key={i} segment={seg} index={i} />
      ))}

      {/* Actions */}
      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => navigation.navigate("Home")}
        >
          <Text style={styles.primaryButtonText}>New Analysis</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F0F9FF",
  },
  content: {
    padding: 20,
  },
  filename: {
    fontSize: 13,
    color: "#64748B",
    marginBottom: 16,
    textAlign: "center",
  },
  overallCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 16,
    padding: 24,
    alignItems: "center",
    marginBottom: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 3,
  },
  statusBanner: {
    marginTop: 16,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    width: "100%",
  },
  statusNormal: {
    backgroundColor: "#D1FAE5",
    borderLeftWidth: 4,
    borderLeftColor: "#0891B2",
  },
  statusAbnormal: {
    backgroundColor: "#FEF3C7",
    borderLeftWidth: 4,
    borderLeftColor: "#F97316",
  },
  statusText: {
    fontSize: 13,
    fontWeight: "600",
  },
  statusTextNormal: {
    color: "#065F46",
  },
  statusTextAbnormal: {
    color: "#92400E",
  },
  metricsRow: {
    flexDirection: "row",
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#0C4A6E",
    marginBottom: 12,
  },
  distributionRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginBottom: 24,
    gap: 12,
  },
  distItem: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FFFFFF",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  distDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 6,
  },
  distLabel: {
    fontSize: 13,
    color: "#0C4A6E",
    fontWeight: "600",
    marginRight: 8,
  },
  distCount: {
    fontSize: 13,
    color: "#64748B",
    fontWeight: "500",
  },
  actions: {
    alignItems: "center",
    marginTop: 24,
    marginBottom: 40,
  },
  primaryButton: {
    backgroundColor: "#0C4A6E",
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 40,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 16,
  },
});
