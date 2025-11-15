const { useState, useEffect } = React;

// Lucide icon components
const AlertCircle = ({ className }) => React.createElement('svg', { className, fill: "none", viewBox: "0 0 24 24", stroke: "currentColor" },
  React.createElement('path', { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" })
);

const TrendingUp = ({ className }) => React.createElement('svg', { className, fill: "none", viewBox: "0 0 24 24", stroke: "currentColor" },
  React.createElement('path', { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" })
);

const Network = ({ className }) => React.createElement('svg', { className, fill: "none", viewBox: "0 0 24 24", stroke: "currentColor" },
  React.createElement('circle', { cx: "12", cy: "12", r: "3", strokeWidth: 2 }),
  React.createElement('circle', { cx: "6", cy: "6", r: "2", strokeWidth: 2 }),
  React.createElement('circle', { cx: "18", cy: "6", r: "2", strokeWidth: 2 }),
  React.createElement('circle', { cx: "6", cy: "18", r: "2", strokeWidth: 2 }),
  React.createElement('circle', { cx: "18", cy: "18", r: "2", strokeWidth: 2 }),
  React.createElement('path', { strokeLinecap: "round", strokeWidth: 2, d: "M9 9l6 6M9 15l6-6" })
);

const FileText = ({ className }) => React.createElement('svg', { className, fill: "none", viewBox: "0 0 24 24", stroke: "currentColor" },
  React.createElement('path', { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" })
);

const Play = ({ className }) => React.createElement('svg', { className, fill: "none", viewBox: "0 0 24 24", stroke: "currentColor" },
  React.createElement('path', { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" }),
  React.createElement('path', { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M21 12a9 9 0 11-18 0 9 9 0 0118 0z" })
);

const RefreshCw = ({ className }) => React.createElement('svg', { className, fill: "none", viewBox: "0 0 24 24", stroke: "currentColor" },
  React.createElement('path', { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" })
);

const AMLMonitoringSystem = () => {
  const [transactionGraph, setTransactionGraph] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [selectedCase, setSelectedCase] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const generateTransactionGraph = () => {
    const entities = [
      { id: 'E1', name: 'Account_Alpha', type: 'personal', risk: 0.2 },
      { id: 'E2', name: 'ShellCo_Beta', type: 'shell', risk: 0.85 },
      { id: 'E3', name: 'Account_Gamma', type: 'personal', risk: 0.3 },
      { id: 'E4', name: 'OffshoreHolding_Delta', type: 'shell', risk: 0.92 },
      { id: 'E5', name: 'Account_Epsilon', type: 'personal', risk: 0.15 },
      { id: 'E6', name: 'TradingCo_Zeta', type: 'business', risk: 0.45 },
      { id: 'E7', name: 'Account_Theta', type: 'personal', risk: 0.75 },
      { id: 'E8', name: 'CryptoExchange_Iota', type: 'business', risk: 0.65 },
    ];

    const transactions = [
      { from: 'E1', to: 'E2', amount: 50000, timestamp: '2024-01-15', suspicious: true },
      { from: 'E2', to: 'E4', amount: 48500, timestamp: '2024-01-16', suspicious: true },
      { from: 'E4', to: 'E6', amount: 47000, timestamp: '2024-01-17', suspicious: true },
      { from: 'E3', to: 'E2', amount: 35000, timestamp: '2024-01-18', suspicious: true },
      { from: 'E5', to: 'E6', amount: 15000, timestamp: '2024-01-19', suspicious: false },
      { from: 'E6', to: 'E7', amount: 25000, timestamp: '2024-01-20', suspicious: true },
      { from: 'E7', to: 'E8', amount: 24000, timestamp: '2024-01-21', suspicious: true },
      { from: 'E1', to: 'E3', amount: 8000, timestamp: '2024-01-22', suspicious: false },
    ];

    return { entities, transactions };
  };

  const detectAnomalies = (graph) => {
    const detected = [];
    
    const pattern1 = {
      id: 'A1',
      type: 'Layering Pattern',
      severity: 'High',
      riskScore: 0.89,
      entities: ['E1', 'E2', 'E4', 'E6'],
      totalAmount: 145500,
      hops: 3,
      description: 'Fund layering through shell companies',
      indicators: [
        'Sequential transactions within 72 hours',
        'Multiple shell company intermediaries',
        'Amount decay pattern (2-3% per hop)',
        'Cross-border entities involved'
      ],
      explanation: {
        graphStructure: 'Linear chain with high-risk nodes',
        temporalPattern: 'Rapid succession (24-48hr intervals)',
        amountPattern: 'Structured to avoid reporting thresholds',
        entityRisk: 'Shell companies with no legitimate business'
      }
    };

    const pattern2 = {
      id: 'A2',
      type: 'Smurfing & Aggregation',
      severity: 'Medium',
      riskScore: 0.72,
      entities: ['E3', 'E2', 'E7', 'E8'],
      totalAmount: 84000,
      hops: 3,
      description: 'Multiple small deposits aggregated and moved',
      indicators: [
        'Multiple source accounts',
        'Aggregation at intermediary',
        'Final movement to crypto exchange',
        'Amounts below reporting threshold'
      ],
      explanation: {
        graphStructure: 'Convergence pattern (many-to-one)',
        temporalPattern: 'Coordinated timing',
        amountPattern: 'Structuring to avoid CTR filing',
        entityRisk: 'Crypto exchange with weak KYC'
      }
    };

    detected.push(pattern1, pattern2);
    return detected;
  };

  const runAnalysis = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      const graph = generateTransactionGraph();
      setTransactionGraph(graph);
      const detected = detectAnomalies(graph);
      setAnomalies(detected);
      setIsAnalyzing(false);
      setSelectedCase(detected[0]);
    }, 1500);
  };

  useEffect(() => {
    runAnalysis();
  }, []);

  const generateReport = (anomaly) => {
    return `
ANTI-MONEY LAUNDERING CASE REPORT
═══════════════════════════════════════════════

Case ID: ${anomaly.id}
Classification: ${anomaly.type}
Risk Score: ${(anomaly.riskScore * 100).toFixed(1)}%
Severity: ${anomaly.severity}

EXECUTIVE SUMMARY
─────────────────
${anomaly.description}

Total Transaction Value: $${anomaly.totalAmount.toLocaleString()}
Number of Hops: ${anomaly.hops}
Entities Involved: ${anomaly.entities.length}

DETECTION INDICATORS
────────────────────
${anomaly.indicators.map((ind, i) => `${i + 1}. ${ind}`).join('\n')}

GRAPH NEURAL NETWORK ANALYSIS
──────────────────────────────
Graph Structure: ${anomaly.explanation.graphStructure}
Temporal Pattern: ${anomaly.explanation.temporalPattern}
Amount Pattern: ${anomaly.explanation.amountPattern}
Entity Risk: ${anomaly.explanation.entityRisk}

INVOLVED ENTITIES
─────────────────
${transactionGraph.entities
  .filter(e => anomaly.entities.includes(e.id))
  .map(e => `${e.id}: ${e.name} (${e.type}) - Risk: ${(e.risk * 100).toFixed(0)}%`)
  .join('\n')}

RECOMMENDATION
──────────────
Escalate for manual review and file Suspicious Activity Report (SAR).
Recommend enhanced due diligence on all involved entities.

Analyst: Arjun Laxman
Organization: Pennsylvania State University
Generated: ${new Date().toLocaleString()}
System: AML Multi-Hop Entity Resolution v2.0
    `.trim();
  };

  const downloadReport = () => {
    if (!selectedCase) return;
    const report = generateReport(selectedCase);
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AML_Report_${selectedCase.id}_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return React.createElement('div', { className: "min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6" },
    React.createElement('div', { className: "max-w-7xl mx-auto" },
      React.createElement('div', { className: "bg-slate-800/50 backdrop-blur-sm border border-blue-500/30 rounded-lg p-6 mb-6" },
        React.createElement('div', { className: "flex items-center justify-between flex-wrap gap-4" },
          React.createElement('div', null,
            React.createElement('h1', { className: "text-3xl font-bold text-white mb-2" }, "Anti-Money Laundering Monitoring System"),
            React.createElement('p', { className: "text-blue-300" }, "Multi-Hop Entity Resolution using Graph Neural Networks"),
            React.createElement('p', { className: "text-gray-400 text-sm mt-1" }, "Developed by Arjun Laxman | Pennsylvania State University")
          ),
          React.createElement('button', {
            onClick: runAnalysis,
            disabled: isAnalyzing,
            className: "flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
          },
            isAnalyzing ? 
              React.createElement(React.Fragment, null,
                React.createElement(RefreshCw, { className: "w-5 h-5 animate-spin" }),
                "Analyzing..."
              ) :
              React.createElement(React.Fragment, null,
                React.createElement(Play, { className: "w-5 h-5" }),
                "Run Analysis"
              )
          )
        )
      ),
      React.createElement('div', { className: "grid grid-cols-1 lg:grid-cols-3 gap-6" },
        React.createElement('div', { className: "lg:col-span-1 space-y-4" },
          React.createElement('div', { className: "bg-slate-800/50 backdrop-blur-sm border border-blue-500/30 rounded-lg p-4" },
            React.createElement('h2', { className: "text-xl font-semibold text-white mb-4 flex items-center gap-2" },
              React.createElement(AlertCircle, { className: "w-5 h-5 text-red-400" }),
              "Detected Anomalies"
            ),
            React.createElement('div', { className: "space-y-3" },
              anomalies.map((anomaly) =>
                React.createElement('div', {
                  key: anomaly.id,
                  onClick: () => setSelectedCase(anomaly),
                  className: `p-4 rounded-lg cursor-pointer transition-all ${
                    selectedCase?.id === anomaly.id
                      ? 'bg-blue-600 border-2 border-blue-400'
                      : 'bg-slate-700/50 border border-slate-600 hover:bg-slate-700'
                  }`
                },
                  React.createElement('div', { className: "flex items-start justify-between mb-2" },
                    React.createElement('span', { className: "text-white font-semibold" }, anomaly.type),
                    React.createElement('span', {
                      className: `px-2 py-1 rounded text-xs font-bold ${
                        anomaly.severity === 'High'
                          ? 'bg-red-500 text-white'
                          : 'bg-yellow-500 text-black'
                      }`
                    }, anomaly.severity)
                  ),
                  React.createElement('div', { className: "text-sm text-gray-300 mb-2" }, anomaly.description),
                  React.createElement('div', { className: "flex items-center justify-between text-xs" },
                    React.createElement('span', { className: "text-gray-400" }, `$${anomaly.totalAmount.toLocaleString()}`),
                    React.createElement('span', { className: "text-red-400 font-semibold" }, `Risk: ${(anomaly.riskScore * 100).toFixed(0)}%`)
                  )
                )
              )
            )
          ),
          React.createElement('div', { className: "bg-slate-800/50 backdrop-blur-sm border border-blue-500/30 rounded-lg p-4" },
            React.createElement('h3', { className: "text-lg font-semibold text-white mb-3" }, "System Stats"),
            React.createElement('div', { className: "space-y-2 text-sm" },
              React.createElement('div', { className: "flex justify-between text-gray-300" },
                React.createElement('span', null, "Entities Monitored:"),
                React.createElement('span', { className: "text-white font-semibold" }, transactionGraph?.entities.length || 0)
              ),
              React.createElement('div', { className: "flex justify-between text-gray-300" },
                React.createElement('span', null, "Transactions Analyzed:"),
                React.createElement('span', { className: "text-white font-semibold" }, transactionGraph?.transactions.length || 0)
              ),
              React.createElement('div', { className: "flex justify-between text-gray-300" },
                React.createElement('span', null, "Anomalies Detected:"),
                React.createElement('span', { className: "text-red-400 font-semibold" }, anomalies.length)
              )
            )
          )
        ),
        React.createElement('div', { className: "lg:col-span-2" },
          selectedCase ?
            React.createElement('div', { className: "bg-slate-800/50 backdrop-blur-sm border border-blue-500/30 rounded-lg p-6" },
              React.createElement('div', { className: "flex items-start justify-between mb-6 flex-wrap gap-4" },
                React.createElement('div', null,
                  React.createElement('h2', { className: "text-2xl font-bold text-white mb-2" }, selectedCase.type),
                  React.createElement('p', { className: "text-gray-300" }, selectedCase.description)
                ),
                React.createElement('button', {
                  onClick: downloadReport,
                  className: "flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors"
                },
                  React.createElement(FileText, { className: "w-5 h-5" }),
                  "Export Report"
                )
              ),
              React.createElement('div', { className: "grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6" },
                React.createElement('div', { className: "bg-slate-700/50 rounded-lg p-4" },
                  React.createElement('div', { className: "text-gray-400 text-sm mb-1" }, "Risk Score"),
                  React.createElement('div', { className: "text-2xl font-bold text-red-400" }, `${(selectedCase.riskScore * 100).toFixed(1)}%`)
                ),
                React.createElement('div', { className: "bg-slate-700/50 rounded-lg p-4" },
                  React.createElement('div', { className: "text-gray-400 text-sm mb-1" }, "Total Amount"),
                  React.createElement('div', { className: "text-2xl font-bold text-white" }, `$${selectedCase.totalAmount.toLocaleString()}`)
                ),
                React.createElement('div', { className: "bg-slate-700/50 rounded-lg p-4" },
                  React.createElement('div', { className: "text-gray-400 text-sm mb-1" }, "Transaction Hops"),
                  React.createElement('div', { className: "text-2xl font-bold text-blue-400" }, selectedCase.hops)
                )
              ),
              React.createElement('div', { className: "mb-6" },
                React.createElement('h3', { className: "text-lg font-semibold text-white mb-3 flex items-center gap-2" },
                  React.createElement(TrendingUp, { className: "w-5 h-5 text-yellow-400" }),
                  "Detection Indicators"
                ),
                React.createElement('div', { className: "space-y-2" },
                  selectedCase.indicators.map((indicator, idx) =>
                    React.createElement('div', {
                      key: idx,
                      className: "bg-slate-700/30 border border-slate-600 rounded p-3 text-gray-300"
                    },
                      React.createElement('span', { className: "text-yellow-400 font-semibold mr-2" }, "•"),
                      indicator
                    )
                  )
                )
              ),
              React.createElement('div', { className: "mb-6" },
                React.createElement('h3', { className: "text-lg font-semibold text-white mb-3 flex items-center gap-2" },
                  React.createElement(Network, { className: "w-5 h-5 text-purple-400" }),
                  "Graph Neural Network Analysis"
                ),
                React.createElement('div', { className: "grid grid-cols-1 sm:grid-cols-2 gap-4" },
                  Object.entries(selectedCase.explanation).map(([key, value]) =>
                    React.createElement('div', {
                      key: key,
                      className: "bg-gradient-to-br from-purple-900/30 to-blue-900/30 border border-purple-500/30 rounded-lg p-4"
                    },
                      React.createElement('div', { className: "text-purple-300 text-sm font-semibold mb-1" },
                        key.replace(/([A-Z])/g, ' $1').trim()
                      ),
                      React.createElement('div', { className: "text-white" }, value)
                    )
                  )
                )
              ),
              React.createElement('div', null,
                React.createElement('h3', { className: "text-lg font-semibold text-white mb-3" }, "Involved Entities"),
                React.createElement('div', { className: "space-y-2" },
                  transactionGraph?.entities
                    .filter((e) => selectedCase.entities.includes(e.id))
                    .map((entity) =>
                      React.createElement('div', {
                        key: entity.id,
                        className: "bg-slate-700/30 border border-slate-600 rounded-lg p-4 flex items-center justify-between"
                      },
                        React.createElement('div', null,
                          React.createElement('div', { className: "text-white font-semibold" }, entity.name),
                          React.createElement('div', { className: "text-gray-400 text-sm capitalize" }, entity.type)
                        ),
                        React.createElement('div', { className: "text-right" },
                          React.createElement('div', { className: "text-red-400 font-semibold" }, `${(entity.risk * 100).toFixed(0)}% Risk`),
                          React.createElement('div', {
                            className: `text-xs px-2 py-1 rounded mt-1 ${
                              entity.risk > 0.7
                                ? 'bg-red-500/20 text-red-300'
                                : entity.risk > 0.4
                                ? 'bg-yellow-500/20 text-yellow-300'
                                : 'bg-green-500/20 text-green-300'
                            }`
                          },
                            entity.risk > 0.7 ? 'High Risk' : entity.risk > 0.4 ? 'Medium Risk' : 'Low Risk'
                          )
                        )
                      )
                    )
                )
              )
            ) :
            React.createElement('div', { className: "bg-slate-800/50 backdrop-blur-sm border border-blue-500/30 rounded-lg p-12 text-center" },
              React.createElement(Network, { className: "w-16 h-16 text-gray-600 mx-auto mb-4" }),
              React.createElement('p', { className: "text-gray-400" }, "Select an anomaly to view details")
            )
        )
      )
    )
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(
  React.createElement(AMLMonitoringSystem)
);
