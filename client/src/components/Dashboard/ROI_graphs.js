import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart, PointElement, LineElement } from 'chart.js';
import './ROI_graphs.css';

Chart.register(PointElement, LineElement);

const ROI_graphs = () => {
  const [graph, setGraph] = useState(null);
  const [graph1, setGraph1] = useState(null);
  const [graph2, setGraph2] = useState(null);
  const [graph3, setGraph3] = useState(null);
  const [graph4, setGraph4] = useState(null);
  const [graph5, setGraph5] = useState(null);
  const [graph6, setGraph6] = useState(null);
  const [graph7, setGraph7] = useState(null);
  const [graph8, setGraph8] = useState(null);
  const [graph9, setGraph9] = useState(null);

  useEffect(() => {
    fetchGraphData();
  }, []);

  const fetchGraphData = async () => {
    try {
      const response = await fetch('https://roibackend.shaktilab.org/roi-graphs', {
        method: 'POST',
        headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
            "X-Requested-With": "XMLHttpRequest" // This header is sometimes required by cors-anywhere
        }
      });
      const data = await response.json();
      console.log("F1 Score - Logistic Regression:", data.f1_score_lg);
console.log("F1 Score - Naive Bayes:", data.f1_score_nb);
console.log("F1 Score - Random Forest:", data.f1_score_rf);
console.log("F1 Score - Support Vector Machine:", data.f1_score_svc);
console.log("F1 Score - Decision Tree:", data.f1_score_dt);
console.log("ROI - Logistic Regression:", data.roi_lg);
console.log("ROI - Naive Bayes:", data.roi_nb);
console.log("ROI - Random Forest:", data.roi_rf);
console.log("ROI - Support Vector Machine:", data.roi_svc);
console.log("ROI - Decision Tree:", data.roi_dt);

      setGraph(data.f1_score_lg);
      setGraph1(data.f1_score_nb);
      setGraph2(data.f1_score_rf);
      setGraph3(data.f1_score_svc);
      setGraph4(data.f1_score_dt);
      setGraph5(data.roi_lg);
      setGraph6(data.roi_nb);
      setGraph7(data.roi_rf);
      setGraph8(data.roi_svc);
      setGraph9(data.roi_dt);
    } catch (error) {
      console.error('Error fetching graph data:', error);
    }
};


const renderGraph = (graphData1, graphData2, f1_score, roi, sizeArray) => {
  const chartData = {
    labels: sizeArray,
    datasets: [
      {
        label: f1_score,
        data: graphData1,
        backgroundColor: '#ff7f78',
        borderColor: '#ff7f78',
        borderWidth: 1,
        yAxisID: 'y',
      },
      {
        label: roi,
        data: graphData2,
        backgroundColor: '#AFD88D',
        borderColor: '#AFD88D',
        borderWidth: 1,
        yAxisID: 'y1', 
      },
    ],
  };
  
  const chartOptions = {
    scales: {
      x: {
        title: {
          display: true,
          text: 'Relative Training Size %',
        },
      },
      y: {
        title: {
          display: true,
          text: 'F1 Score',
        },
        position: 'left', 
        suggestedMin: 0.5, 
        suggestedMax: 0.7, 
        id: 'y',
      },
      y1: {
        title: {
          display: true,
          text: 'ROI',
        },
        position: 'right', 
        suggestedMin: 0,  
        suggestedMax: 18, 
        id: 'y1',
      },
    },
  };
  
  return (
      <Line data={chartData} options={chartOptions} />
  );
};

  const sizeArray = [20, 30, 40, 50, 60, 70, 80, 90];

  return (
      <div className="chart-container section">
        
        <div className="chart-row subsection">
          <div className="chart-wrapper">
            {renderGraph(graph, graph5, 'F1 score', 'ROI', sizeArray)}
            F1 score vs ROI vs Relative Training Size - Logistic Regression
          </div>
        </div>
        <div className="chart-row subsection">
          <div className="chart-wrapper">
            {renderGraph(graph1, graph6, 'F1 score', 'ROI', sizeArray)}
            F1 score vs ROI vs Relative Training Size - Naive Bayes
          </div>
        </div>
        <div className="chart-row subsection">
          <div className="chart-wrapper">
            {renderGraph(graph2, graph7, 'F1 score', 'ROI', sizeArray)}
            F1 score vs ROI vs Relative Training Size - Random Forest
          </div>
        </div>
        <div className="chart-row subsection">
          <div className="chart-wrapper">
            {renderGraph(graph3, graph8, 'F1 score', 'ROI', sizeArray)}
            F1 score vs ROI vs Relative Training Size - SVC
          </div>
        </div>
        <div className="chart-row subsection">
          <div className="chart-wrapper">
            {renderGraph(graph4, graph9, 'F1 score', 'ROI', sizeArray)}
            F1 score vs ROI vs Relative Training Size - Decision Tree
          </div>
        </div>
      </div>
  );
};

export default ROI_graphs;
