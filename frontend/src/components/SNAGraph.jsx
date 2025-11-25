import React, { useEffect, useRef } from "react";
import CytoscapeComponent from "react-cytoscapejs";

const SNAGraph = ({ elements }) => {
  const cyRef = useRef(null);

  // Re-run layout ONLY when elements change
  useEffect(() => {
    if (!cyRef.current) return;

    try {
      const cy = cyRef.current;

      // Prevent crash by stopping previous layout
      cy.stop();

      cy.json({ elements });

      cy.layout({
        name: "cose",
        animate: true,
        fit: true,
        padding: 30
      }).run();
      
    } catch (err) {
      console.error("Cytoscape render error:", err);
    }
  }, [elements]);

  return (
    <CytoscapeComponent
      cy={(cy) => (cyRef.current = cy)}
      elements={[]}
      style={{ width: "100%", height: "600px" }}
      stylesheet={[
        {
          selector: "node",
          style: {
            label: "data(id)",
            "font-size": 12,
            "background-color": "#667eea",
            color: "#fff"
          },
        },
        {
          selector: "edge",
          style: {
            width: "mapData(weight, 1, 1000, 1, 12)",
            "line-color": "#764ba2",
            opacity: 0.8,
          },
        },
      ]}
    />
  );
};

export default SNAGraph;
