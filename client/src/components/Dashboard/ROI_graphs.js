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
      const response = await fetch('http://44.201.124.234:5000/roi-graphs', {
        method: 'POST',
      });
      const data = await response.json();
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
          backgroundColor: '#DAF0F7',
          borderColor: '#DAF0F7',
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: roi,
          data: graphData2,
          backgroundColor: '#C8D9F0',
          borderColor: '#C8D9F0',
          borderWidth: 1,
          pointRadius: 0,
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
          suggestedMin: 0,
          suggestedMax: 1,
          id: 'left-axis', 
        },
        y1: {
          title: {
            display: true,
            text: 'ROI',
          },
          position: 'right',
          suggestedMin: 0,
          suggestedMax: 100,
          id: 'right-axis', 
        },
      },
      elements: {
        point: {
          radius: 0,
        },
      },
    };

    return (
      <div>
        <Line data={chartData} options={chartOptions} />
      </div>
    );
  };

  const sizeArray = [20, 30, 40, 50, 60, 70, 80, 90];

  return (
    <div className="chart-container">
      <div className="chart-row">
        <div className="chart-wrapper">
          {renderGraph(graph, graph5, 'F1 score', 'ROI', sizeArray)}
          <br />
          F1 score vs ROI vs Relative Training Size - Logistic Regression
        </div>
        <div className="chart-wrapper">
          {renderGraph(graph1, graph6, 'F1 score', 'ROI', sizeArray)}
          <br />
          F1 score vs ROI vs Relative Training Size - Naive Bayes
        </div>
      </div>
      <div className="chart-row">
        <div className="chart-wrapper">
          {renderGraph(graph2, graph7, 'F1 score', 'ROI', sizeArray)}
          <br />
          F1 score vs ROI vs Relative Training Size - Random Forest
        </div>
        <div className="chart-wrapper">
          {renderGraph(graph3, graph8, 'F1 score', 'ROI', sizeArray)}
          <br />
          F1 score vs ROI vs Relative Training Size - SVC
        </div>
      </div>
      <div className="chart-row">
        <div className="chart-wrapper">
          {renderGraph(graph4, graph9, 'F1 score', 'ROI', sizeArray)}
          <br />
          F1 score vs ROI vs Relative Training Size - Decision Tree
        </div>
      </div>
    </div>
  );
};
export default ROI_graphs;
