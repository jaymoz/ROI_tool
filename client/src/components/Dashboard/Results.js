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
      const response = await fetch('http://44.201.124.234:5000/f1score', {
        method: 'POST',
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

  const renderGraphs = (graphData1, graphData2, graphData3, graphData4, graphData5, lg, nb, rf, svc, dt, sizeArray) => {

    const chartData = {
      labels: sizeArray,
      datasets: [
        {
          label: lg,
          data: graphData1,
          backgroundColor: '#DAF0F7',
          borderColor: '#DAF0F7',
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: nb,
          data: graphData2,
          backgroundColor: '#C8D9F0',
          borderColor: '#C8D9F0', 
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: rf,
          data: graphData3,
          backgroundColor: '#A8B5E0',
          borderColor: '#A8B5E0', 
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: svc,
          data: graphData4,
          backgroundColor: '#A8B5E0',
          borderColor: '#A8B5E0', 
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: dt,
          data: graphData5,
          backgroundColor: '#C3EEFA',
          borderColor: '#C3EEFA', 
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
      <center><div className="text">Results</div><br/><br/></center>
      <div className="chartwrapper">{renderGraphs(graph, graph3, graph6, graph7, graph8, 'Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVC', 'Decision Tree', sizeArray)}</div>
      <div className="chartwrapper" style={{height: '100px', marginTop:'0px'}}>F1 score vs Relative Training Size %</div>
      <br/><br/>
      <div className="chartwrapper">{renderGraphs(graph1, graph4, graph9, graph10, graph11, 'Logistic Regression', 'Naive Bayes','Random Forest', 'SVC', 'Decision Tree', sizeArray)}</div>
      <div className="chartwrapper" style={{height: '100px', marginTop:'0px'}}>Recall score vs Relative Training Size %</div>
      <br/><br/>
      <div className="chartwrapper">{renderGraphs(graph2, graph5, graph12, graph13, graph14, 'Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVC', 'Decision Tree', sizeArray)}</div>
      <div className="chartwrapper">Precision score vs Relative Training Size %</div>
      <br/><br/>
    </div>
  );
};

export default Results;
