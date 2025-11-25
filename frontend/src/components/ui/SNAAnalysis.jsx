import React, { useEffect, useState } from "react";
import axios from "axios";
import CytoscapeComponent from "react-cytoscapejs";
import { API } from "../../api/config";


const SNAAnalysis = ({ uploadId }) => {
    const [graph, setGraph] = useState(null);

    useEffect(() => {
        axios.get(`${API}/sna/${uploadId}`)
            .then(res => setGraph(res.data))
            .catch(err => console.error(err));
    }, [uploadId]);

    const elements = React.useMemo(() => {
        if (!graph || !graph.nodes || !graph.edges) return [];
        return [
            ...graph.nodes.map(n => ({
                data: { id: String(n.id), label: `${n.id} (${n.degree})` }
            })),
            ...graph.edges.map(e => ({
                data: { source: String(e.source), target: String(e.target), weight: e.weight }
            }))
        ];
    }, [graph]);

    if (!graph) return <div className="p-4">Loading analisis jaringan...</div>;

    if (!graph.sna_enabled)
        return (
            <div className="p-4 text-red-500 font-medium">
                Tidak ada data hubungan (A-number ↔ B-number), SNA tidak tersedia.
            </div>
        );

    return (
        <div className="p-4">
            <h2 className="text-2xl font-bold mb-4">Analisis Social Network</h2>

            <CytoscapeComponent
                key={uploadId + (graph ? graph.nodes.length : 0)}
                elements={elements}
                style={{ width: "100%", height: "600px", borderRadius: "12px", border: "1px solid #eee" }}
                layout={{
                    name: "cose",
                    animate: false,
                    nodeDimensionsIncludeLabels: true,
                    randomize: true,
                    idealEdgeLength: 100,
                    nodeOverlap: 20,
                    refresh: 20,
                    fit: true,
                    padding: 30,
                    componentSpacing: 100,
                    nodeRepulsion: 400000,
                    edgeElasticity: 100,
                    nestingFactor: 5,
                }}
                stylesheet={[
                    {
                        selector: 'node',
                        style: {
                            'background-color': '#667eea',
                            'label': 'data(label)',
                            'color': '#333',
                            'font-size': '10px',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'width': '40px',
                            'height': '40px'
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 2,
                            'line-color': '#ccc',
                            'target-arrow-color': '#ccc',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier'
                        }
                    }
                ]}
            />

            <h3 className="text-xl font-semibold mt-6">Statistik Jaringan</h3>
            <pre className="bg-gray-100 p-4 rounded mt-2 text-sm overflow-auto">
                {JSON.stringify(graph.stats, null, 2)}
            </pre>
        </div>
    );
};

export default SNAAnalysis;
