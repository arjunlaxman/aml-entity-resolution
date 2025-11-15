const { useState, useEffect, useRef, useMemo, useCallback } = React;

// Enhanced Icon Components with animations
const IconWrapper = ({ children, className = "", pulse = false, spin = false }) => {
  const animationClass = pulse ? "animate-pulse" : spin ? "animate-spin" : "";
  return React.createElement('div', { className: `${className} ${animationClass}` }, children);
};

// Enhanced Icons
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

const Shield = ({ className }) => React.createElement('svg', { className, fill: "none", viewBox: "0 0 24 24", stroke: "currentColor" },
  React.createElement('path', { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" })
);

const Brain = ({ className }) => React.createElement('svg', { className, fill: "none", viewBox: "0 0 24 24", stroke: "currentColor" },
  React.createElement('path', { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" })
);

const Activity = ({ className }) => React.createElement('svg', { className, fill: "none", viewBox: "0 0 24 24", stroke: "currentColor" },
  React.createElement('polyline', { points: "22 12 18 12 15 21 9 3 6 12 2 12", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2 })
);

const Globe = ({ className }) => React.createElement('svg', { className, fill: "none", viewBox: "0 0 24 24", stroke: "currentColor" },
  React.createElement('circle', { cx: "12", cy: "12", r: "10", strokeWidth: 2 }),
  React.createElement('line', { x1: "2", y1: "12", x2: "22", y2: "12", strokeWidth: 2 }),
  React.createElement('path', { strokeWidth: 2, d: "M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" })
);

// Advanced Components
const RiskMeter = ({ risk }) => {
  const angle = (risk * 180) - 90;
  const color = risk > 0.7 ? '#ef4444' : risk > 0.4 ? '#eab308' : '#10b981';
  
  return React.createElement('div', { className: "relative w-32 h-32" },
    React.createElement('svg', { viewBox: "0 0 200 120", className: "w-full h-full" },
      // Background arc
      React.createElement('path', { 
        d: "M 30 100 A 70 70 0 0 1 170 100",
        fill: "none",
        stroke: "#1e293b",
        strokeWidth: "15"
      }),
      // Risk arc
      React.createElement('path', { 
        d: "M 30 100 A 70 70 0 0 1 170 100",
        fill: "none",
        stroke: color,
        strokeWidth: "15",
        strokeDasharray: `${risk * 220} 220`,
        className: "transition-all duration-1000"
      }),
      // Needle
      React.createElement('line', {
        x1: "100",
        y1: "100",
        x2: 100 + 60 * Math.cos(angle * Math.PI / 180),
        y2: 100 + 60 * Math.sin(angle * Math.PI / 180),
        stroke: "#ffffff",
        strokeWidth: "3",
        strokeLinecap: "round",
        className: "transition-all duration-1000"
      }),
      React.createElement('circle', { cx: "100", cy: "100", r: "8", fill: "#ffffff" })
    ),
    React.createElement('div', { className: "absolute inset-0 flex items-center justify-center top-4" },
      React.createElement('div', { className: "text-center" },
        React.createElement('div', { className: "text-3xl font-bold", style: { color } }, `${(risk * 100).toFixed(0)}%`),
        React.createElement('div', { className: "text-xs text-gray-400 uppercase tracking-wider" }, "Risk Score")
      )
    )
  );
};

const StatCard = ({ icon: Icon, title, value, color, trend, subtitle }) => {
  return React.createElement('div', { 
    className: "bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 hover:shadow-2xl transition-all duration-300 hover:scale-[1.02]" 
  },
    React.createElement('div', { className: "flex items-start justify-between mb-4" },
      React.createElement('div', { className: `p-3 rounded-xl bg-gradient-to-br ${color}` },
        React.createElement(Icon, { className: "w-6 h-6 text-white" })
      ),
      trend && React.createElement('div', { 
        className: `text-sm px-2 py-1 rounded-full ${trend > 0 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}` 
      }, `${trend > 0 ? '↑' : '↓'} ${Math.abs(trend)}%`)
    ),
    React.createElement('div', { className: "text-3xl font-bold text-white mb-1" }, value),
    React.createElement('div', { className: "text-sm text-gray-400" }, title),
    subtitle && React.createElement('div', { className: "text-xs text-gray-500 mt-2" }, subtitle)
  );
};

const NetworkVisualization = ({ entities, transactions, selectedCase }) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) - 50;
    
    // Animation frame
    let animationId;
    let time = 0;
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw connections with animated flow
      transactions.forEach((transaction, idx) => {
        const fromEntity = entities.find(e => e.id === transaction.from);
        const toEntity = entities.find(e => e.id === transaction.to);
        if (!fromEntity || !toEntity) return;
        
        const fromIdx = entities.indexOf(fromEntity);
        const toIdx = entities.indexOf(toEntity);
        const fromAngle = (fromIdx / entities.length) * Math.PI * 2;
        const toAngle = (toIdx / entities.length) * Math.PI * 2;
        
        const fromX = centerX + radius * Math.cos(fromAngle);
        const fromY = centerY + radius * Math.sin(fromAngle);
        const toX = centerX + radius * Math.cos(toAngle);
        const toY = centerY + radius * Math.sin(toAngle);
        
        // Draw connection line
        ctx.strokeStyle = transaction.suspicious ? 'rgba(239, 68, 68, 0.6)' : 'rgba(59, 130, 246, 0.3)';
        ctx.lineWidth = transaction.suspicious ? 3 : 1;
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        
        // Add curve for better visualization
        const controlX = centerX + (Math.cos((fromAngle + toAngle) / 2) * radius * 0.6);
        const controlY = centerY + (Math.sin((fromAngle + toAngle) / 2) * radius * 0.6);
        ctx.quadraticCurveTo(controlX, controlY, toX, toY);
        ctx.stroke();
        
        // Animated particles along path for suspicious transactions
        if (transaction.suspicious) {
          const particleProgress = ((time + idx * 20) % 100) / 100;
          const particleX = (1 - particleProgress) * (1 - particleProgress) * fromX + 
                           2 * (1 - particleProgress) * particleProgress * controlX + 
                           particleProgress * particleProgress * toX;
          const particleY = (1 - particleProgress) * (1 - particleProgress) * fromY + 
                           2 * (1 - particleProgress) * particleProgress * controlY + 
                           particleProgress * particleProgress * toY;
          
          ctx.fillStyle = 'rgba(239, 68, 68, 1)';
          ctx.beginPath();
          ctx.arc(particleX, particleY, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      });
      
      // Draw entities
      entities.forEach((entity, idx) => {
        const angle = (idx / entities.length) * Math.PI * 2;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        
        // Highlight selected entities
        const isSelected = selectedCase && selectedCase.entities.includes(entity.id);
        
        // Entity circle with gradient
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 20);
        if (entity.type === 'shell') {
          gradient.addColorStop(0, 'rgba(239, 68, 68, 0.9)');
          gradient.addColorStop(1, 'rgba(239, 68, 68, 0.3)');
        } else if (entity.type === 'business') {
          gradient.addColorStop(0, 'rgba(168, 85, 247, 0.9)');
          gradient.addColorStop(1, 'rgba(168, 85, 247, 0.3)');
        } else {
          gradient.addColorStop(0, 'rgba(59, 130, 246, 0.9)');
          gradient.addColorStop(1, 'rgba(59, 130, 246, 0.3)');
        }
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, isSelected ? 25 : 20, 0, Math.PI * 2);
        ctx.fill();
        
        if (isSelected) {
          // Pulsing ring for selected entities
          ctx.strokeStyle = 'rgba(251, 191, 36, ' + (0.5 + 0.3 * Math.sin(time / 10)) + ')';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(x, y, 30 + 5 * Math.sin(time / 10), 0, Math.PI * 2);
          ctx.stroke();
        }
        
        // Entity label
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(entity.name.split('_')[0], x, y + 40);
        
        // Risk indicator
        if (entity.risk > 0.7) {
          ctx.fillStyle = '#ef4444';
          ctx.beginPath();
          ctx.arc(x + 15, y - 15, 5, 0, Math.PI * 2);
          ctx.fill();
        }
      });
      
      // Center logo/label
      ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.font = 'bold 24px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('GNN ANALYSIS', centerX, centerY);
      
      time++;
      animationId = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [entities, transactions, selectedCase]);
  
  return React.createElement('canvas', { 
    ref: canvasRef,
    className: "w-full h-full",
    style: { minHeight: '400px' }
  });
};

const FeatureImportanceChart = ({ features }) => {
  const maxValue = Math.max(...features.map(f => f.value));
  
  return React.createElement('div', { className: "space-y-3" },
    features.map((feature, idx) => 
      React.createElement('div', { key: idx, className: "relative" },
        React.createElement('div', { className: "flex justify-between text-sm mb-1" },
          React.createElement('span', { className: "text-gray-300" }, feature.name),
          React.createElement('span', { className: "text-white font-semibold" }, feature.value.toFixed(3))
        ),
        React.createElement('div', { className: "h-2 bg-slate-700 rounded-full overflow-hidden" },
          React.createElement('div', { 
            className: `h-full rounded-full transition-all duration-1000 ${
              feature.impact === 'high' ? 'bg-gradient-to-r from-red-600 to-red-400' :
              feature.impact === 'medium' ? 'bg-gradient-to-r from-yellow-600 to-yellow-400' :
              'bg-gradient-to-r from-blue-600 to-blue-400'
            }`,
            style: { width: `${(feature.value / maxValue) * 100}%` }
          })
        )
      )
    )
  );
};

const TimelineView = ({ transactions, entities }) => {
  return React.createElement('div', { className: "relative" },
    React.createElement('div', { className: "absolute left-4 top-0 bottom-0 w-0.5 bg-gradient-to-b from-blue-500 to-purple-500" }),
    React.createElement('div', { className: "space-y-6 pl-12" },
      transactions.map((transaction, idx) => {
        const fromEntity = entities.find(e => e.id === transaction.from);
        const toEntity = entities.find(e => e.id === transaction.to);
        
        return React.createElement('div', { key: idx, className: "relative" },
          React.createElement('div', { 
            className: `absolute -left-8 w-4 h-4 rounded-full ${
              transaction.suspicious ? 'bg-red-500' : 'bg-blue-500'
            } ring-4 ring-slate-800`
          }),
          React.createElement('div', { 
            className: `p-4 rounded-lg ${
              transaction.suspicious 
                ? 'bg-gradient-to-r from-red-900/30 to-orange-900/30 border border-red-500/30' 
                : 'bg-slate-800/50 border border-slate-700'
            }`
          },
            React.createElement('div', { className: "flex items-start justify-between" },
              React.createElement('div', null,
                React.createElement('div', { className: "text-xs text-gray-400 mb-1" }, transaction.timestamp),
                React.createElement('div', { className: "text-white font-semibold" },
                  `${fromEntity?.name} → ${toEntity?.name}`
                ),
                React.createElement('div', { className: "text-2xl font-bold text-green-400 mt-1" },
                  `$${transaction.amount.toLocaleString()}`
                )
              ),
              transaction.suspicious && React.createElement('div', { 
                className: "px-3 py-1 bg-red-500/20 text-red-400 rounded-full text-xs font-semibold animate-pulse" 
              }, "SUSPICIOUS")
            )
          )
        );
      })
    )
  );
};

// Main Enhanced AML Component
const EnhancedAMLSystem = () => {
  const [activeView, setActiveView] = useState('overview');
  const [transactionGraph, setTransactionGraph] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [selectedCase, setSelectedCase] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [realTimeAlerts, setRealTimeAlerts] = useState([]);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [filterRisk, setFilterRisk] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  // Enhanced data generation with more complexity
  const generateEnhancedTransactionGraph = () => {
    const entities = [
      { id: 'E1', name: 'Account_Alpha', type: 'personal', risk: 0.22, country: 'US', kyc: 0.95 },
      { id: 'E2', name: 'ShellCo_Beta', type: 'shell', risk: 0.89, country: 'KY', kyc: 0.12 },
      { id: 'E3', name: 'Account_Gamma', type: 'personal', risk: 0.31, country: 'US', kyc: 0.88 },
      { id: 'E4', name: 'OffshoreHolding_Delta', type: 'shell', risk: 0.94, country: 'PA', kyc: 0.08 },
      { id: 'E5', name: 'Account_Epsilon', type: 'personal', risk: 0.15, country: 'UK', kyc: 0.92 },
      { id: 'E6', name: 'TradingCo_Zeta', type: 'business', risk: 0.48, country: 'SG', kyc: 0.75 },
      { id: 'E7', name: 'Account_Theta', type: 'personal', risk: 0.76, country: 'RU', kyc: 0.34 },
      { id: 'E8', name: 'CryptoExchange_Iota', type: 'business', risk: 0.68, country: 'MT', kyc: 0.55 },
      { id: 'E9', name: 'InvestmentFund_Kappa', type: 'business', risk: 0.35, country: 'US', kyc: 0.90 },
      { id: 'E10', name: 'ShellCo_Lambda', type: 'shell', risk: 0.91, country: 'VG', kyc: 0.05 },
    ];

    const transactions = [
      { from: 'E1', to: 'E2', amount: 50000, timestamp: '2024-01-15 09:30:00', suspicious: true, method: 'wire' },
      { from: 'E2', to: 'E4', amount: 48500, timestamp: '2024-01-16 14:22:00', suspicious: true, method: 'swift' },
      { from: 'E4', to: 'E6', amount: 47000, timestamp: '2024-01-17 11:45:00', suspicious: true, method: 'wire' },
      { from: 'E3', to: 'E2', amount: 35000, timestamp: '2024-01-18 16:10:00', suspicious: true, method: 'ach' },
      { from: 'E5', to: 'E6', amount: 15000, timestamp: '2024-01-19 10:05:00', suspicious: false, method: 'wire' },
      { from: 'E6', to: 'E7', amount: 25000, timestamp: '2024-01-20 13:30:00', suspicious: true, method: 'crypto' },
      { from: 'E7', to: 'E8', amount: 24000, timestamp: '2024-01-21 08:45:00', suspicious: true, method: 'crypto' },
      { from: 'E1', to: 'E3', amount: 8000, timestamp: '2024-01-22 12:00:00', suspicious: false, method: 'ach' },
      { from: 'E9', to: 'E10', amount: 100000, timestamp: '2024-01-23 15:20:00', suspicious: true, method: 'swift' },
      { from: 'E10', to: 'E8', amount: 98000, timestamp: '2024-01-24 09:15:00', suspicious: true, method: 'crypto' },
    ];

    return { entities, transactions };
  };

  const detectEnhancedAnomalies = (graph) => {
    const detected = [];
    
    const pattern1 = {
      id: 'A1',
      type: 'Complex Layering Scheme',
      severity: 'Critical',
      riskScore: 0.93,
      confidence: 0.87,
      entities: ['E1', 'E2', 'E4', 'E6'],
      totalAmount: 145500,
      hops: 3,
      description: 'Sophisticated fund layering through multiple shell companies',
      indicators: [
        'Sequential transactions within 72 hours',
        'Multiple shell company intermediaries in high-risk jurisdictions',
        'Consistent amount decay pattern (2-3% per hop)',
        'Cross-border entities with low KYC scores',
        'Wire transfers followed by crypto conversion'
      ],
      mlExplanation: {
        graphStructure: 'Linear chain with 3 high-risk nodes (risk > 0.8)',
        temporalPattern: 'Rapid succession with 24-48hr intervals',
        amountPattern: 'Structured below $50k CTR threshold',
        entityRisk: 'Shell companies (E2, E4) with KYC < 0.15',
        geoRisk: 'Funds routed through KY, PA (high-risk jurisdictions)'
      },
      shapValues: [
        { name: 'Shell Company Involvement', value: 0.342, impact: 'high' },
        { name: 'Transaction Velocity', value: 0.287, impact: 'high' },
        { name: 'Geographic Risk Score', value: 0.198, impact: 'medium' },
        { name: 'Amount Decay Pattern', value: 0.173, impact: 'medium' }
      ]
    };

    const pattern2 = {
      id: 'A2',
      type: 'Smurfing & Crypto Conversion',
      severity: 'High',
      riskScore: 0.78,
      confidence: 0.92,
      entities: ['E3', 'E2', 'E7', 'E8'],
      totalAmount: 84000,
      hops: 3,
      description: 'Structured deposits aggregated and converted to cryptocurrency',
      indicators: [
        'Multiple source accounts below threshold',
        'Aggregation at shell company intermediary',
        'Final conversion to cryptocurrency',
        'Endpoint is crypto exchange with weak KYC',
        'Russian entity involvement'
      ],
      mlExplanation: {
        graphStructure: 'Convergence pattern (many-to-one-to-many)',
        temporalPattern: 'Coordinated timing across 4 days',
        amountPattern: 'All transactions below $40k reporting limit',
        entityRisk: 'Crypto exchange (E8) with KYC score 0.55',
        geoRisk: 'Involvement of RU entity (sanctions risk)'
      },
      shapValues: [
        { name: 'Crypto Conversion', value: 0.298, impact: 'high' },
        { name: 'Structuring Pattern', value: 0.234, impact: 'high' },
        { name: 'Sanctions Risk', value: 0.189, impact: 'medium' },
        { name: 'Convergence Score', value: 0.145, impact: 'low' }
      ]
    };

    const pattern3 = {
      id: 'A3',
      type: 'Trade-Based Laundering',
      severity: 'High',
      riskScore: 0.85,
      confidence: 0.79,
      entities: ['E9', 'E10', 'E8'],
      totalAmount: 198000,
      hops: 2,
      description: 'Large fund movement through offshore shell to crypto',
      indicators: [
        'Investment fund to shell company transfer',
        'Offshore shell in British Virgin Islands',
        'Immediate conversion to cryptocurrency',
        'Minimal value decay (2%)',
        'Transaction amounts above $75k threshold'
      ],
      mlExplanation: {
        graphStructure: 'Direct path through highest-risk entities',
        temporalPattern: 'Rapid 48-hour completion',
        amountPattern: 'Large amounts triggering SAR requirements',
        entityRisk: 'Shell company (E10) with KYC 0.05',
        geoRisk: 'BVI shell company (tax haven indicator)'
      },
      shapValues: [
        { name: 'Transaction Amount', value: 0.312, impact: 'high' },
        { name: 'Offshore Involvement', value: 0.276, impact: 'high' },
        { name: 'Speed of Transfer', value: 0.201, impact: 'medium' },
        { name: 'Entity Type Mismatch', value: 0.156, impact: 'low' }
      ]
    };

    detected.push(pattern1, pattern2, pattern3);
    return detected;
  };

  const generateModelMetrics = () => {
    return {
      accuracy: 0.912,
      precision: 0.893,
      recall: 0.876,
      f1Score: 0.884,
      auc: 0.947,
      falsePositiveRate: 0.041,
      processedToday: 14287,
      flaggedToday: 43,
      avgProcessingTime: 187,
      modelVersion: '3.2.1',
      lastTraining: '2024-01-10',
      dataPoints: 2847293
    };
  };

  const generateRealTimeAlerts = () => {
    return [
      { id: 'RT1', time: '2 min ago', type: 'High Risk', message: 'Suspicious pattern detected: $125,000 transfer to high-risk jurisdiction' },
      { id: 'RT2', time: '5 min ago', type: 'New Entity', message: 'Unknown shell company registered in Cayman Islands' },
      { id: 'RT3', time: '12 min ago', type: 'Velocity', message: 'Rapid transactions detected: 5 transfers in 30 minutes' },
      { id: 'RT4', time: '18 min ago', type: 'Threshold', message: 'CTR threshold approached: $9,950 structured deposit' },
    ];
  };

  const runEnhancedAnalysis = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      const graph = generateEnhancedTransactionGraph();
      setTransactionGraph(graph);
      const detected = detectEnhancedAnomalies(graph);
      setAnomalies(detected);
      setModelMetrics(generateModelMetrics());
      setRealTimeAlerts(generateRealTimeAlerts());
      setIsAnalyzing(false);
      setSelectedCase(detected[0]);
    }, 2000);
  };

  useEffect(() => {
    runEnhancedAnalysis();
  }, []);

  const downloadReport = () => {
    const report = generateDetailedReport(selectedCase, transactionGraph, modelMetrics);
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AML_Report_${selectedCase.id}_${new Date().toISOString()}.txt`;
    a.click();
  };

  const generateDetailedReport = (anomaly, graph, metrics) => {
    return `
╔══════════════════════════════════════════════════════════════════════╗
║          ANTI-MONEY LAUNDERING INVESTIGATION REPORT                  ║
║                   CONFIDENTIAL - LAW ENFORCEMENT USE ONLY           ║
╚══════════════════════════════════════════════════════════════════════╝

REPORT METADATA
═══════════════════════════════════════════════════════════════════════
Report ID:            ${anomaly.id}-${Date.now()}
Generated:            ${new Date().toISOString()}
System Version:       ${metrics.modelVersion}
Classification:       SUSPICIOUS ACTIVITY REPORT (SAR)
Jurisdiction:         Multiple (Cross-Border)
Regulatory Framework: FATF / FinCEN / EU 5AMLD

EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════
Case Type:            ${anomaly.type}
Risk Assessment:      ${anomaly.severity.toUpperCase()} (${(anomaly.riskScore * 100).toFixed(1)}%)
ML Confidence:        ${(anomaly.confidence * 100).toFixed(1)}%
Total Amount:         $${anomaly.totalAmount.toLocaleString()} USD
Transaction Hops:     ${anomaly.hops}
Entities Involved:    ${anomaly.entities.length}

Primary Findings:
${anomaly.description}

THREAT INDICATORS
═══════════════════════════════════════════════════════════════════════
${anomaly.indicators.map((ind, idx) => `  ${idx + 1}. ${ind}`).join('\n')}

GRAPH NEURAL NETWORK ANALYSIS
═══════════════════════════════════════════════════════════════════════
Model Architecture:   3-Layer Graph Attention Network (GAT)
Node Embeddings:      128-dimensional
Attention Heads:      8
Training Dataset:     ${metrics.dataPoints.toLocaleString()} labeled transactions

Pattern Detection Results:
${Object.entries(anomaly.mlExplanation).map(([key, value]) => 
  `  • ${key.replace(/([A-Z])/g, ' $1').trim()}: ${value}`
).join('\n')}

EXPLAINABLE AI - SHAP FEATURE IMPORTANCE
═══════════════════════════════════════════════════════════════════════
Top Contributing Features:
${anomaly.shapValues.map((feature, idx) => 
  `  ${idx + 1}. ${feature.name}: ${feature.value.toFixed(3)} [${feature.impact.toUpperCase()}]`
).join('\n')}

ENTITY ANALYSIS
═══════════════════════════════════════════════════════════════════════
${anomaly.entities.map(entityId => {
  const entity = graph.entities.find(e => e.id === entityId);
  return `
Entity ID:     ${entity.id}
Name:          ${entity.name}
Type:          ${entity.type.toUpperCase()}
Risk Score:    ${(entity.risk * 100).toFixed(0)}%
Jurisdiction:  ${entity.country}
KYC Score:     ${(entity.kyc * 100).toFixed(0)}%
─────────────────────────────────────────────────────────────`;
}).join('\n')}

TRANSACTION FLOW ANALYSIS
═══════════════════════════════════════════════════════════════════════
${graph.transactions
  .filter(t => anomaly.entities.includes(t.from) || anomaly.entities.includes(t.to))
  .map(t => {
    const fromEntity = graph.entities.find(e => e.id === t.from);
    const toEntity = graph.entities.find(e => e.id === t.to);
    return `
Timestamp:     ${t.timestamp}
From:          ${fromEntity.name} (${fromEntity.country})
To:            ${toEntity.name} (${toEntity.country})
Amount:        $${t.amount.toLocaleString()} USD
Method:        ${t.method.toUpperCase()}
Suspicious:    ${t.suspicious ? 'YES ⚠️' : 'NO'}
─────────────────────────────────────────────────────────────`;
  }).join('\n')}

MODEL PERFORMANCE METRICS
═══════════════════════════════════════════════════════════════════════
Accuracy:             ${(metrics.accuracy * 100).toFixed(1)}%
Precision:            ${(metrics.precision * 100).toFixed(1)}%
Recall:               ${(metrics.recall * 100).toFixed(1)}%
F1-Score:             ${(metrics.f1Score * 100).toFixed(1)}%
AUC-ROC:              ${metrics.auc.toFixed(3)}
False Positive Rate:  ${(metrics.falsePositiveRate * 100).toFixed(1)}%

Processing Statistics (24hr):
  • Transactions Analyzed: ${metrics.processedToday.toLocaleString()}
  • Anomalies Detected:    ${metrics.flaggedToday}
  • Avg Processing Time:    ${metrics.avgProcessingTime}ms

RECOMMENDED ACTIONS
═══════════════════════════════════════════════════════════════════════
1. IMMEDIATE: File Suspicious Activity Report (SAR) with FinCEN
2. IMMEDIATE: Freeze accounts associated with shell companies
3. HIGH: Initiate enhanced due diligence on all involved entities
4. HIGH: Contact cross-border financial intelligence units
5. MEDIUM: Review all transactions from past 90 days for these entities
6. MEDIUM: Update risk scoring models with identified patterns

REGULATORY COMPLIANCE
═══════════════════════════════════════════════════════════════════════
☑ Bank Secrecy Act (BSA) - 31 USC 5311
☑ USA PATRIOT Act - Section 314(a) Information Sharing
☑ FATF Recommendation 16 - Wire Transfer Requirements
☑ EU 5th AML Directive - Article 30 & 31
☑ OFAC Sanctions Screening - Completed

CASE OFFICER NOTES
═══════════════════════════════════════════════════════════════════════
[Auto-generated by ML System - Requires Human Review]
This case exhibits classic money laundering typologies with high 
confidence indicators. The involvement of shell companies in high-risk
jurisdictions combined with rapid fund movement and crypto conversion
strongly suggests intentional obfuscation of fund origin.

Priority escalation recommended due to:
- Amount exceeds $100,000 threshold
- Multiple international jurisdictions
- Cryptocurrency involvement
- Shell company red flags

═══════════════════════════════════════════════════════════════════════
END OF REPORT - DISTRIBUTION RESTRICTED
Generated by: AML-GNN System v${metrics.modelVersion}
Powered by: PyTorch Geometric + Deep Graph Library
═══════════════════════════════════════════════════════════════════════
`;
  };

  const filteredAnomalies = useMemo(() => {
    return anomalies.filter(a => {
      const matchesRisk = filterRisk === 'all' || 
                         (filterRisk === 'high' && a.riskScore > 0.7) ||
                         (filterRisk === 'medium' && a.riskScore > 0.4 && a.riskScore <= 0.7) ||
                         (filterRisk === 'low' && a.riskScore <= 0.4);
      const matchesSearch = searchTerm === '' || 
                           a.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           a.description.toLowerCase().includes(searchTerm.toLowerCase());
      return matchesRisk && matchesSearch;
    });
  }, [anomalies, filterRisk, searchTerm]);

  return React.createElement('div', { className: "min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900" },
    // Animated background
    React.createElement('div', { className: "fixed inset-0 opacity-20" },
      React.createElement('div', { 
        className: "absolute inset-0",
        style: {
          backgroundImage: `radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.3) 0%, transparent 50%),
                           radial-gradient(circle at 80% 80%, rgba(168, 85, 247, 0.3) 0%, transparent 50%),
                           radial-gradient(circle at 40% 20%, rgba(239, 68, 68, 0.2) 0%, transparent 50%)`
        }
      })
    ),
    
    // Main Content
    React.createElement('div', { className: "relative z-10" },
      // Header
      React.createElement('header', { className: "bg-slate-900/80 backdrop-blur-xl border-b border-slate-700/50" },
        React.createElement('div', { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8" },
          React.createElement('div', { className: "flex items-center justify-between py-6" },
            React.createElement('div', { className: "flex items-center gap-4" },
              React.createElement(Shield, { className: "w-10 h-10 text-blue-500" }),
              React.createElement('div', null,
                React.createElement('h1', { className: "text-3xl font-bold text-white" }, "AML Neural Monitor"),
                React.createElement('p', { className: "text-sm text-gray-400" }, "Graph Neural Network Detection System v3.2.1")
              )
            ),
            React.createElement('div', { className: "flex items-center gap-4" },
              React.createElement('button', {
                onClick: runEnhancedAnalysis,
                disabled: isAnalyzing,
                className: "flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-semibold transition-all transform hover:scale-105"
              },
                isAnalyzing ? 
                  React.createElement(RefreshCw, { className: "w-5 h-5 animate-spin" }) :
                  React.createElement(Play, { className: "w-5 h-5" }),
                isAnalyzing ? "Analyzing..." : "Run Analysis"
              ),
              React.createElement('div', { className: "flex items-center gap-2 px-4 py-2 bg-slate-800 rounded-lg" },
                React.createElement('div', { className: `w-3 h-3 rounded-full ${isAnalyzing ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}` }),
                React.createElement('span', { className: "text-sm text-gray-300" }, 
                  isAnalyzing ? 'Processing' : 'System Active'
                )
              )
            )
          )
        )
      ),
      
      // Navigation Tabs
      React.createElement('div', { className: "bg-slate-800/50 backdrop-blur-sm border-b border-slate-700/50" },
        React.createElement('div', { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8" },
          React.createElement('div', { className: "flex gap-6 py-4 overflow-x-auto" },
            ['overview', 'network', 'timeline', 'models', 'alerts'].map(view =>
              React.createElement('button', {
                key: view,
                onClick: () => setActiveView(view),
                className: `px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
                  activeView === view 
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white' 
                    : 'text-gray-400 hover:text-white hover:bg-slate-700'
                }`
              }, view.charAt(0).toUpperCase() + view.slice(1))
            )
          )
        )
      ),
      
      // Main Content Area
      React.createElement('div', { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" },
        // Stats Cards
        activeView === 'overview' && React.createElement('div', { className: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8" },
          React.createElement(StatCard, {
            icon: Activity,
            title: "Transactions Today",
            value: modelMetrics?.processedToday.toLocaleString() || '0',
            color: "from-blue-600 to-cyan-600",
            trend: 12,
            subtitle: "↑ 12% from yesterday"
          }),
          React.createElement(StatCard, {
            icon: AlertCircle,
            title: "Anomalies Detected",
            value: modelMetrics?.flaggedToday || '0',
            color: "from-red-600 to-orange-600",
            trend: -8,
            subtitle: "↓ 8% improvement"
          }),
          React.createElement(StatCard, {
            icon: Brain,
            title: "Model Accuracy",
            value: modelMetrics ? `${(modelMetrics.accuracy * 100).toFixed(1)}%` : '0%',
            color: "from-purple-600 to-pink-600",
            subtitle: "F1: " + (modelMetrics?.f1Score.toFixed(3) || '0')
          }),
          React.createElement(StatCard, {
            icon: Globe,
            title: "Jurisdictions",
            value: "12",
            color: "from-green-600 to-teal-600",
            subtitle: "3 high-risk flagged"
          })
        ),
        
        // Filter Bar
        React.createElement('div', { className: "bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-4 mb-6" },
          React.createElement('div', { className: "flex flex-wrap items-center gap-4" },
            React.createElement('input', {
              type: "text",
              placeholder: "Search anomalies...",
              value: searchTerm,
              onChange: (e) => setSearchTerm(e.target.value),
              className: "flex-1 min-w-[200px] bg-slate-900 border border-slate-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            }),
            React.createElement('select', {
              value: filterRisk,
              onChange: (e) => setFilterRisk(e.target.value),
              className: "bg-slate-900 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            },
              React.createElement('option', { value: 'all' }, 'All Risk Levels'),
              React.createElement('option', { value: 'high' }, 'High Risk Only'),
              React.createElement('option', { value: 'medium' }, 'Medium Risk'),
              React.createElement('option', { value: 'low' }, 'Low Risk')
            ),
            React.createElement('div', { className: "text-sm text-gray-400" },
              `Showing ${filteredAnomalies.length} of ${anomalies.length} anomalies`
            )
          )
        ),
        
        // Main Grid Layout
        React.createElement('div', { className: "grid grid-cols-1 lg:grid-cols-3 gap-6" },
          // Left Panel - Anomaly List
          React.createElement('div', { className: "lg:col-span-1 space-y-4" },
            React.createElement('div', { className: "bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl border border-slate-700/50 rounded-xl p-6" },
              React.createElement('h2', { className: "text-xl font-bold text-white mb-4 flex items-center gap-2" },
                React.createElement(AlertCircle, { className: "w-5 h-5 text-red-400" }),
                "Detected Anomalies"
              ),
              React.createElement('div', { className: "space-y-3 max-h-[600px] overflow-y-auto pr-2" },
                filteredAnomalies.map((anomaly) =>
                  React.createElement('div', {
                    key: anomaly.id,
                    onClick: () => setSelectedCase(anomaly),
                    className: `p-4 rounded-xl cursor-pointer transition-all transform hover:scale-[1.02] ${
                      selectedCase?.id === anomaly.id
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 shadow-lg ring-2 ring-blue-400'
                        : 'bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700'
                    }`
                  },
                    React.createElement('div', { className: "flex items-start justify-between mb-2" },
                      React.createElement('div', null,
                        React.createElement('div', { className: "font-bold text-white" }, anomaly.type),
                        React.createElement('div', { className: "text-xs text-gray-400 mt-1" }, `ID: ${anomaly.id}`)
                      ),
                      React.createElement('span', {
                        className: `px-3 py-1 rounded-full text-xs font-bold ${
                          anomaly.severity === 'Critical' ? 'bg-red-500 text-white animate-pulse' :
                          anomaly.severity === 'High' ? 'bg-orange-500 text-white' :
                          'bg-yellow-500 text-black'
                        }`
                      }, anomaly.severity)
                    ),
                    React.createElement('div', { className: "text-sm text-gray-300 mb-3" }, anomaly.description),
                    React.createElement('div', { className: "flex items-center justify-between" },
                      React.createElement('div', { className: "text-xs space-y-1" },
                        React.createElement('div', { className: "text-gray-400" }, 
                          React.createElement('span', { className: "text-green-400 font-semibold" }, `$${anomaly.totalAmount.toLocaleString()}`)
                        ),
                        React.createElement('div', { className: "text-gray-400" }, 
                          `${anomaly.hops} hops • ${anomaly.entities.length} entities`
                        )
                      ),
                      React.createElement(RiskMeter, { risk: anomaly.riskScore })
                    ),
                    React.createElement('div', { className: "mt-3 pt-3 border-t border-slate-700" },
                      React.createElement('div', { className: "flex items-center justify-between text-xs" },
                        React.createElement('span', { className: "text-gray-400" }, "ML Confidence:"),
                        React.createElement('div', { className: "flex items-center gap-2" },
                          React.createElement('div', { className: "w-20 h-1.5 bg-slate-700 rounded-full overflow-hidden" },
                            React.createElement('div', { 
                              className: "h-full bg-gradient-to-r from-blue-500 to-purple-500",
                              style: { width: `${anomaly.confidence * 100}%` }
                            })
                          ),
                          React.createElement('span', { className: "text-white font-semibold" }, 
                            `${(anomaly.confidence * 100).toFixed(0)}%`
                          )
                        )
                      )
                    )
                  )
                )
              )
            ),
            
            // Real-time Alerts
            activeView === 'overview' && React.createElement('div', { className: "bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl border border-slate-700/50 rounded-xl p-6" },
              React.createElement('h3', { className: "text-lg font-bold text-white mb-4 flex items-center gap-2" },
                React.createElement(Activity, { className: "w-5 h-5 text-yellow-400 animate-pulse" }),
                "Real-time Alerts"
              ),
              React.createElement('div', { className: "space-y-3" },
                realTimeAlerts.map(alert =>
                  React.createElement('div', {
                    key: alert.id,
                    className: "bg-slate-800/50 border-l-4 border-yellow-500 rounded-lg p-3"
                  },
                    React.createElement('div', { className: "flex items-start justify-between" },
                      React.createElement('div', null,
                        React.createElement('div', { className: "text-yellow-400 font-semibold text-sm" }, alert.type),
                        React.createElement('div', { className: "text-gray-300 text-xs mt-1" }, alert.message)
                      ),
                      React.createElement('div', { className: "text-gray-500 text-xs whitespace-nowrap" }, alert.time)
                    )
                  )
                )
              )
            )
          ),
          
          // Right Panel - Case Details
          React.createElement('div', { className: "lg:col-span-2" },
            selectedCase ? React.createElement('div', { className: "bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl border border-slate-700/50 rounded-xl p-6" },
              // Case Header
              React.createElement('div', { className: "flex items-start justify-between mb-6 pb-6 border-b border-slate-700" },
                React.createElement('div', null,
                  React.createElement('div', { className: "flex items-center gap-3 mb-2" },
                    React.createElement('h2', { className: "text-3xl font-bold text-white" }, selectedCase.type),
                    React.createElement('span', {
                      className: `px-3 py-1 rounded-full text-sm font-bold ${
                        selectedCase.severity === 'Critical' ? 'bg-red-500 text-white animate-pulse' :
                        selectedCase.severity === 'High' ? 'bg-orange-500 text-white' :
                        'bg-yellow-500 text-black'
                      }`
                    }, selectedCase.severity)
                  ),
                  React.createElement('p', { className: "text-gray-400 text-lg" }, selectedCase.description)
                ),
                React.createElement('button', {
                  onClick: downloadReport,
                  className: "flex items-center gap-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-6 py-3 rounded-lg font-semibold transition-all transform hover:scale-105 shadow-lg"
                },
                  React.createElement(FileText, { className: "w-5 h-5" }),
                  "Export SAR Report"
                )
              ),
              
              // View-specific content
              activeView === 'overview' && React.createElement('div', { className: "space-y-6" },
                // Metrics Grid
                React.createElement('div', { className: "grid grid-cols-1 sm:grid-cols-3 gap-4" },
                  React.createElement('div', { className: "bg-gradient-to-br from-red-900/30 to-orange-900/30 border border-red-500/30 rounded-xl p-4" },
                    React.createElement('div', { className: "text-gray-400 text-sm mb-1" }, "Risk Score"),
                    React.createElement('div', { className: "text-3xl font-bold text-red-400" }, `${(selectedCase.riskScore * 100).toFixed(1)}%`),
                    React.createElement('div', { className: "text-xs text-gray-500 mt-1" }, "Critical threshold > 70%")
                  ),
                  React.createElement('div', { className: "bg-gradient-to-br from-green-900/30 to-emerald-900/30 border border-green-500/30 rounded-xl p-4" },
                    React.createElement('div', { className: "text-gray-400 text-sm mb-1" }, "Total Amount"),
                    React.createElement('div', { className: "text-3xl font-bold text-green-400" }, `$${selectedCase.totalAmount.toLocaleString()}`),
                    React.createElement('div', { className: "text-xs text-gray-500 mt-1" }, "Across all transactions")
                  ),
                  React.createElement('div', { className: "bg-gradient-to-br from-blue-900/30 to-purple-900/30 border border-blue-500/30 rounded-xl p-4" },
                    React.createElement('div', { className: "text-gray-400 text-sm mb-1" }, "Network Depth"),
                    React.createElement('div', { className: "text-3xl font-bold text-blue-400" }, `${selectedCase.hops} hops`),
                    React.createElement('div', { className: "text-xs text-gray-500 mt-1" }, `${selectedCase.entities.length} entities involved`)
                  )
                ),
                
                // Detection Indicators
                React.createElement('div', null,
                  React.createElement('h3', { className: "text-lg font-bold text-white mb-3 flex items-center gap-2" },
                    React.createElement(TrendingUp, { className: "w-5 h-5 text-yellow-400" }),
                    "Detection Indicators"
                  ),
                  React.createElement('div', { className: "grid grid-cols-1 sm:grid-cols-2 gap-3" },
                    selectedCase.indicators.map((indicator, idx) =>
                      React.createElement('div', {
                        key: idx,
                        className: "bg-slate-800/50 border border-slate-700 rounded-lg p-3 flex items-start gap-2"
                      },
                        React.createElement('span', { className: "text-yellow-400 text-lg" }, "⚠"),
                        React.createElement('span', { className: "text-gray-300 text-sm" }, indicator)
                      )
                    )
                  )
                ),
                
                // ML Explanation
                React.createElement('div', null,
                  React.createElement('h3', { className: "text-lg font-bold text-white mb-3 flex items-center gap-2" },
                    React.createElement(Brain, { className: "w-5 h-5 text-purple-400" }),
                    "Graph Neural Network Analysis"
                  ),
                  React.createElement('div', { className: "grid grid-cols-1 sm:grid-cols-2 gap-4" },
                    Object.entries(selectedCase.mlExplanation).map(([key, value]) =>
                      React.createElement('div', {
                        key: key,
                        className: "bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-500/30 rounded-lg p-4"
                      },
                        React.createElement('div', { className: "text-purple-300 text-sm font-semibold mb-2" },
                          key.replace(/([A-Z])/g, ' $1').trim()
                        ),
                        React.createElement('div', { className: "text-white text-sm" }, value)
                      )
                    )
                  )
                ),
                
                // SHAP Values
                React.createElement('div', null,
                  React.createElement('h3', { className: "text-lg font-bold text-white mb-3 flex items-center gap-2" },
                    React.createElement(Activity, { className: "w-5 h-5 text-green-400" }),
                    "Feature Importance (SHAP)"
                  ),
                  React.createElement('div', { className: "bg-slate-800/50 rounded-lg p-4" },
                    React.createElement(FeatureImportanceChart, { features: selectedCase.shapValues })
                  )
                )
              ),
              
              // Network View
              activeView === 'network' && React.createElement('div', { className: "space-y-6" },
                React.createElement('div', { className: "bg-slate-800/50 rounded-xl p-4 h-[500px]" },
                  React.createElement(NetworkVisualization, {
                    entities: transactionGraph?.entities || [],
                    transactions: transactionGraph?.transactions || [],
                    selectedCase
                  })
                ),
                React.createElement('div', { className: "grid grid-cols-3 gap-4 text-center" },
                  React.createElement('div', { className: "bg-slate-800/50 rounded-lg p-4" },
                    React.createElement('div', { className: "flex items-center justify-center gap-2 mb-2" },
                      React.createElement('div', { className: "w-3 h-3 rounded-full bg-blue-500" }),
                      React.createElement('span', { className: "text-gray-300 text-sm" }, "Personal Account")
                    )
                  ),
                  React.createElement('div', { className: "bg-slate-800/50 rounded-lg p-4" },
                    React.createElement('div', { className: "flex items-center justify-center gap-2 mb-2" },
                      React.createElement('div', { className: "w-3 h-3 rounded-full bg-red-500" }),
                      React.createElement('span', { className: "text-gray-300 text-sm" }, "Shell Company")
                    )
                  ),
                  React.createElement('div', { className: "bg-slate-800/50 rounded-lg p-4" },
                    React.createElement('div', { className: "flex items-center justify-center gap-2 mb-2" },
                      React.createElement('div', { className: "w-3 h-3 rounded-full bg-purple-500" }),
                      React.createElement('span', { className: "text-gray-300 text-sm" }, "Business Entity")
                    )
                  )
                )
              ),
              
              // Timeline View
              activeView === 'timeline' && React.createElement('div', { className: "space-y-6" },
                React.createElement('h3', { className: "text-lg font-bold text-white mb-4" }, "Transaction Timeline"),
                React.createElement(TimelineView, {
                  transactions: transactionGraph?.transactions || [],
                  entities: transactionGraph?.entities || []
                })
              ),
              
              // Model Metrics View
              activeView === 'models' && modelMetrics && React.createElement('div', { className: "space-y-6" },
                React.createElement('div', { className: "grid grid-cols-2 sm:grid-cols-4 gap-4" },
                  React.createElement('div', { className: "bg-slate-800/50 rounded-lg p-4 text-center" },
                    React.createElement('div', { className: "text-2xl font-bold text-green-400" }, `${(modelMetrics.accuracy * 100).toFixed(1)}%`),
                    React.createElement('div', { className: "text-xs text-gray-400 mt-1" }, "Accuracy")
                  ),
                  React.createElement('div', { className: "bg-slate-800/50 rounded-lg p-4 text-center" },
                    React.createElement('div', { className: "text-2xl font-bold text-blue-400" }, `${(modelMetrics.precision * 100).toFixed(1)}%`),
                    React.createElement('div', { className: "text-xs text-gray-400 mt-1" }, "Precision")
                  ),
                  React.createElement('div', { className: "bg-slate-800/50 rounded-lg p-4 text-center" },
                    React.createElement('div', { className: "text-2xl font-bold text-purple-400" }, `${(modelMetrics.recall * 100).toFixed(1)}%`),
                    React.createElement('div', { className: "text-xs text-gray-400 mt-1" }, "Recall")
                  ),
                  React.createElement('div', { className: "bg-slate-800/50 rounded-lg p-4 text-center" },
                    React.createElement('div', { className: "text-2xl font-bold text-yellow-400" }, modelMetrics.auc.toFixed(3)),
                    React.createElement('div', { className: "text-xs text-gray-400 mt-1" }, "AUC-ROC")
                  )
                ),
                React.createElement('div', { className: "bg-slate-800/50 rounded-lg p-6" },
                  React.createElement('h4', { className: "text-white font-semibold mb-4" }, "Model Information"),
                  React.createElement('div', { className: "space-y-2 text-sm" },
                    React.createElement('div', { className: "flex justify-between text-gray-300" },
                      React.createElement('span', null, "Version:"),
                      React.createElement('span', { className: "text-white font-mono" }, modelMetrics.modelVersion)
                    ),
                    React.createElement('div', { className: "flex justify-between text-gray-300" },
                      React.createElement('span', null, "Last Training:"),
                      React.createElement('span', { className: "text-white" }, modelMetrics.lastTraining)
                    ),
                    React.createElement('div', { className: "flex justify-between text-gray-300" },
                      React.createElement('span', null, "Training Data Points:"),
                      React.createElement('span', { className: "text-white" }, modelMetrics.dataPoints.toLocaleString())
                    ),
                    React.createElement('div', { className: "flex justify-between text-gray-300" },
                      React.createElement('span', null, "Avg Processing Time:"),
                      React.createElement('span', { className: "text-white" }, `${modelMetrics.avgProcessingTime}ms`)
                    )
                  )
                )
              ),
              
              // Alerts View
              activeView === 'alerts' && React.createElement('div', { className: "space-y-4" },
                realTimeAlerts.map(alert =>
                  React.createElement('div', {
                    key: alert.id,
                    className: "bg-slate-800/50 border-l-4 border-yellow-500 rounded-lg p-4"
                  },
                    React.createElement('div', { className: "flex items-start justify-between" },
                      React.createElement('div', null,
                        React.createElement('div', { className: "text-yellow-400 font-semibold" }, alert.type),
                        React.createElement('div', { className: "text-gray-300 mt-1" }, alert.message)
                      ),
                      React.createElement('div', { className: "text-gray-500 text-sm" }, alert.time)
                    )
                  )
                )
              )
            ) : React.createElement('div', { className: "bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl border border-slate-700/50 rounded-xl p-12 text-center" },
              React.createElement(Shield, { className: "w-16 h-16 text-gray-600 mx-auto mb-4" }),
              React.createElement('h3', { className: "text-xl font-semibold text-gray-400 mb-2" }, "No Case Selected"),
              React.createElement('p', { className: "text-gray-500" }, "Select an anomaly from the list to view detailed analysis")
            )
          )
        )
      )
    )
  );
};

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(React.createElement(EnhancedAMLSystem));
