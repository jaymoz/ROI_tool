import React, { useEffect, useState, useRef, forwardRef, useImperativeHandle } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart, PointElement, LineElement } from 'chart.js';
import './Results.css';

Chart.register(PointElement, LineElement);

const Results = forwardRef((props, ref) => {

  

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
  const [graph10, setGraph10] = useState(null);
  const [graph11, setGraph11] = useState(null);
  const [graph12, setGraph12] = useState(null);
  const [graph13, setGraph13] = useState(null);
  const [graph14, setGraph14] = useState(null);

  const resetGraphData = () => {
    setGraph(null);
    setGraph3(null);
    setGraph6(null);
    setGraph7(null);
    setGraph8(null);
    setGraph1(null);
    setGraph4(null);
    setGraph9(null);
    setGraph10(null);
    setGraph11(null);
    setGraph2(null);
    setGraph5(null);
    setGraph12(null);
    setGraph13(null);
    setGraph14(null);
  }

  const fetchGraphData = async () => {
    try {
      const response = await fetch('https://roibackend.shaktilab.org/f1score', {
        method: 'POST',
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
          "X-Requested-With": "XMLHttpRequest"
        }
      });

      const data = await response.json();
      setGraph(data.f1_score_lg);
      setGraph3(data.f1_score_nb);
      setGraph6(data.f1_score_rf);
      setGraph7(data.f1_score_svc);
      setGraph8(data.f1_score_dt);
      setGraph1(data.recall_score_lg);
      setGraph4(data.recall_score_nb);
      setGraph9(data.recall_score_rf);
      setGraph10(data.recall_score_svc);
      setGraph11(data.recall_score_dt);
      setGraph2(data.precision_score_lg);
      setGraph5(data.precision_score_nb);
      setGraph12(data.precision_score_rf);
      setGraph13(data.precision_score_svc);
      setGraph14(data.precision_score_dt);
    } catch (error) {
      console.error('Error fetching graph data:', error);
    }
  };

  useImperativeHandle(
    ref, ()=>{
      return{
        fetchGraphData: fetchGraphData,
        resetGraphData: resetGraphData
      };
    }
  )

  useEffect(() => {
    fetchGraphData();
  }, []);

  const renderGraphs = (graphData1, graphData2, graphData3, graphData4, graphData5, sizeArray) => {
    const datasets = [
      {
        label: 'Logistic Regression',
        data: graphData1,
        backgroundColor: '#FF1205',
        borderColor: '#FF1205',
        borderWidth: 1,
        pointRadius: 0,
      },
      {
        label: 'Naive Bayes',
        data: graphData2,
        backgroundColor: '#77DD77',
        borderColor: '#77DD77',
        borderWidth: 1,
        pointRadius: 0,
      },
      {
        label: 'Random Forest',
        data: graphData3,
        backgroundColor: '#ffd400',
        borderColor: '#ffd400',
        borderWidth: 1,
        pointRadius: 0,
      },
      {
        label: 'SVC',
        data: graphData4,
        backgroundColor: '#0C2D48',
        borderColor: '#0C2D48',
        borderWidth: 1,
        pointRadius: 0,
      },
      {
        label: 'Decision Tree',
        data: graphData5,
        backgroundColor: '#C45AEC',
        borderColor: '#C45AEC',
        borderWidth: 1,
        pointRadius: 0,
      },
    ];

    const chartData = {
      labels: sizeArray,
      datasets: datasets,
    };

    const chartOptions = {
      responsive: true,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Relative Training Size',
          },
        },
        y: {
          title: {
            display: true,
            text: 'Score',
          },
          suggestedMin: 0,
          suggestedMax: 1,
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
    <div className="chartcontainer">
      <div className='section'>
        <div className='graph-name-group subsection'>
          <div className="chartwrapper">{renderGraphs(graph, graph3, graph6, graph7, graph8, sizeArray)}</div>
          <div>F1 score vs Relative Training Size %</div>
        </div>
        <div className='graph-name-group subsection'>
          <div className="chartwrapper">{renderGraphs(graph1, graph4, graph9, graph10, graph11, sizeArray)}</div>
          <div>Recall score vs Relative Training Size %</div>
        </div>
        <div className='graph-name-group subsection'>
          <div className="chartwrapper">{renderGraphs(graph2, graph5, graph12, graph13, graph14, sizeArray)}</div>
          <div>Precision score vs Relative Training Size %</div>
        </div>
      </div>
    </div>
  );
});

export default Results;
