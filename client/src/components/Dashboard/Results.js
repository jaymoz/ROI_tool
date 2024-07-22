import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {Chart, PointElement, LineElement} from 'chart.js';
import './Results.css'

Chart.register(PointElement, LineElement);

const Results = () => {
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


  useEffect(() => {
    fetchGraphData();
  }, []);

  const fetchGraphData = async () => {
    try {
      const response = await fetch('https://roibackend.shaktilab.org/f1score', {
          method: 'POST',
          headers: {
              "Access-Control-Allow-Origin": "*",
              "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
              "X-Requested-With": "XMLHttpRequest" // Required by some versions of cors-anywhere
          }
      });
      
      const data = await response.json();
      // console.log("F1 Score - Logistic Regression:", data.f1_score_lg);
      // console.log("F1 Score - Naive Bayes:", data.f1_score_nb);
      // console.log("F1 Score - Random Forest:", data.f1_score_rf);
      // console.log("F1 Score - Support Vector Machine:", data.f1_score_svc);
      // console.log("F1 Score - Decision Tree:", data.f1_score_dt);
      // console.log("Recall Score - Logistic Regression:", data.recall_score_lg);
      // console.log("Recall Score - Naive Bayes:", data.recall_score_nb);
      // console.log("Recall Score - Random Forest:", data.recall_score_rf);
      // console.log("Recall Score - Support Vector Machine:", data.recall_score_svc);
      // console.log("Recall Score - Decision Tree:", data.recall_score_dt);
      // console.log("Precision Score - Logistic Regression:", data.precision_score_lg);
      // console.log("Precision Score - Naive Bayes:", data.precision_score_nb);
      // console.log("Precision Score - Random Forest:", data.precision_score_rf);
      // console.log("Precision Score - Support Vector Machine:", data.precision_score_svc);
      // console.log("Precision Score - Decision Tree:", data.precision_score_dt);

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

  const renderGraphs = (graphData1, graphData2, graphData3, graphData4, graphData5, lg, nb, rf, svc, dt, sizeArray) => {

    const chartData = {
      labels: sizeArray,
      datasets: [
        {
          label: lg,
          data: graphData1,
          backgroundColor: '#FF1205',
          borderColor: '#FF1205',
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: nb,
          data: graphData2,
          backgroundColor: '#77DD77',
          borderColor: '#77DD77',
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: rf,
          data: graphData3,
          backgroundColor: '#ffd400',
          borderColor: '#ffd400',
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: svc,
          data: graphData4,
          backgroundColor: '#0C2D48',
          borderColor: '#0C2D48',
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: dt,
          data: graphData5,
          backgroundColor: '#C45AEC',
          borderColor: '#C45AEC',
          borderWidth: 1,
          pointRadius: 0,
        },
      ],
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
            <div className="chartwrapper">{renderGraphs(graph, graph3, graph6, graph7, graph8, 'Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVC', 'Decision Tree', sizeArray)}</div>
            <div>F1 score vs Relative Training Size %</div>
          </div>
          <div className='graph-name-group subsection'>
            <div className="chartwrapper">{renderGraphs(graph1, graph4, graph9, graph10, graph11, 'Logistic Regression', 'Naive Bayes','Random Forest', 'SVC', 'Decision Tree', sizeArray)}</div>
            <div>Recall score vs Relative Training Size %</div>
          </div>
          <div className='graph-name-group subsection'>
            <div className="chartwrapper">{renderGraphs(graph2, graph5, graph12, graph13, graph14, 'Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVC', 'Decision Tree', sizeArray)}</div>
            <div>Precision score vs Relative Training Size %</div>
          </div>
        </div>
      </div>
  );
};

export default Results;
