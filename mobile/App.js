import React, { useState, useCallback } from "react";
import { StatusBar } from "expo-status-bar";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { SafeAreaProvider } from "react-native-safe-area-context";

import HomeScreen from "./src/screens/HomeScreen";
import ResultsScreen from "./src/screens/ResultsScreen";
import HistoryScreen from "./src/screens/HistoryScreen";

const Stack = createNativeStackNavigator();

const headerStyle = {
  headerStyle: { backgroundColor: "#0C4A6E" },
  headerTintColor: "#FFFFFF",
  headerTitleStyle: { fontWeight: "700" },
};

export default function App() {
  const [history, setHistory] = useState([]);

  // Memoized wrappers to handle prop injection
  const ResultsWrapper = useCallback(
    (props) => {
      const { results, filename } = props.route.params || {};

      React.useEffect(() => {
        if (results && filename) {
          setHistory((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.filename === filename) return prev;
            return [...prev, { filename, results, time: new Date().toLocaleString() }];
          });
        }
      }, [results, filename]);

      return <ResultsScreen {...props} />;
    },
    []
  );

  const HistoryWrapper = useCallback(
    (props) => (
      <HistoryScreen
        {...props}
        route={{
          ...props.route,
          params: { ...props.route.params, history, setHistory },
        }}
      />
    ),
    [history]
  );

  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <StatusBar style="light" />
        <Stack.Navigator
          initialRouteName="Home"
          screenOptions={headerStyle}
        >
          <Stack.Screen
            name="Home"
            component={HomeScreen}
            options={{ title: "AusculTek" }}
          />
          <Stack.Screen
            name="Results"
            component={ResultsWrapper}
            options={{ title: "Analysis Results" }}
          />
          <Stack.Screen
            name="History"
            component={HistoryWrapper}
            options={{ title: "History" }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}
