import React from "react";
import CytoscapeComponent from "react-cytoscapejs";

const SNAGraph = ({ elements }) => {
    return (
        <CytoscapeComponent
            elements={elements}
            style={{ width: "100%", height: "600px" }}
            layout={{ name: "cose" }}
            stylesheet={[
                {
                    selector: "node",
                    style: {
                        label: "data(id)",
                        "font-size": 12,
                        "background-color": "#667eea",
                        color: "#fff"
                    }
                },
                {
                    selector: "edge",
                    style: {
                        width: "mapData(weight, 1, 1000, 1, 12)",
                        "line-color": "#764ba2",
                        opacity: 0.8
                    }
                }
            ]}
        />
    );
};

export default SNAGraph;
