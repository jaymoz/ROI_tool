import React, { useState } from 'react';
import Dropdown from './GridDropdown';
import ActiveLearning from './ActiveLearning';
import LearningDropdown from './Learning';
import axios from 'axios';
import './MLdropdown.css'

const MLdropdown = ({ onModelSelect, onLearningSelect, trainData, testData }) => {
  const options = [
    { label: 'Logistic Regression', value: 'logistic_regression' },
    { label: 'Naive Bayes', value: 'naive_bayes' },
    { label: 'Random Forest', value: 'random_forest' },
    { label: 'Support Vector Machine', value: 'support_vector_machine' },
    { label: 'Decision Tree', value: 'decision_tree' },
  ];

  const learningAlgo = [
    { label: 'Supervised Model', value: 'supervised' },
    { label: 'Active Learning', value: 'activeLearning' },
  ];

  const [learning, setLearning] = useState(null);
  const [report, setReport] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [stopTime, setStopTime] = useState(null);
  const [f1_score, setF1Score] = useState(null);
  const [graph, setGraph] = useState(false);
  const [confusionMatrix, setConfusionMatrix] = useState(false);
  const [fp, setFP] = useState(false);
  const [fn, setFN] = useState(false);
  const [tp, setTP] = useState(false);

  const handleLearningSelect = async (selectedValue) => {
    console.log('Selected option:', selectedValue);
    onLearningSelect(selectedValue);

    const headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
        "X-Requested-With": "XMLHttpRequest"  // Required by some versions of cors-anywhere
    };

    if (selectedValue === 'supervised') {
      try {
        const response = await axios.post('http://cors-anywhere.herokuapp.com/https://roibackend.shaktilab.org/weekly-supervised', {}, { headers: headers });
        console.log(response.data);

        if (response.data.success) {
          setLearning('supervised');
        }
      } catch (error) {
        console.error(error);
      }
    } else if (selectedValue === 'activeLearning') {
      try {
        const response = await axios.post('http://cors-anywhere.herokuapp.com/https://roibackend.shaktilab.org/active-learning', {}, { headers: headers });
        console.log(response.data);
        if (response.data.success) {
          setLearning('activeLearning');
        }
      } catch (error) {
        console.error(error);
      }
    }
};


const handleOptionSelect = async (selectedValue) => {
  console.log('Selected option:', selectedValue);
  onModelSelect(selectedValue);

  const headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
      "X-Requested-With": "XMLHttpRequest"
  };

  let endpoint = "";
  switch (selectedValue) {
      case 'logistic_regression':
          endpoint = 'logistic-regression';
          break;
      case 'naive_bayes':
          endpoint = 'naive-bayes';
          break;
      case 'random_forest':
          endpoint = 'random-forest';
          break;
      case 'support_vector_machine':
          endpoint = 'support-vector-machine';
          break;
      case 'decision_tree':
          endpoint = 'decision-tree';
          break;
      default:
          return;
  }

  try {
      const response = await axios.post(`http://cors-anywhere.herokuapp.com/https://roibackend.shaktilab.org/${endpoint}`, {}, { headers: headers });
      console.log(response.data);
      if (response.data.success) {
          setReport(response.data.report);
          setAccuracy(response.data.accuracy);
          setStopTime(response.data.stop);
          setGraph(response.data.graph);
          setConfusionMatrix(response.data.cm);
          setF1Score(response.data.f1);
          setFP(response.data.fp);
          setFN(response.data.fn);
          setTP(response.data.tp);
      }
  } catch (error) {
      console.error(error);
  }
};

  return (
      <>
        <div className="text" style={{ fontSize: '38px'}}>
          ML Models<br/><br/>
        </div>
        <div className="ml-dropdown-container">
          <div className="ml-models" style={{color: '#28a9e2'}}>
            {/* learning */}
            <LearningDropdown options={learningAlgo} onSelect={handleLearningSelect} />
            {learning === 'supervised' && (
                /* supervised models and report */
                <React.Fragment>
                  <br/><br/>
                  <div className="text" style={{ fontSize: '35px', textAlign: 'center', color: '#28a9e2' }}>Supervised Models</div><br/><br/>

                  <Dropdown options={options} onSelect={handleOptionSelect} />
                  {report && (
                      <div className="classification-report">
                        <div className="report-content">
                          <div className="report-tile">
                            <div className="report-tile-title">Accuracy</div>
                            <div className="report-tile-value">{accuracy}</div>
                          </div>
                          <div className="report-tile">
                            <div className="report-tile-title">F1 Score</div>
                            <div className="report-tile-value">{f1_score}</div>
                          </div>
                          <div className="report-tile">
                            <div className="report-tile-title">Execution Time</div>
                            <div className="report-tile-value">{stopTime} seconds</div>
                          </div><br/><hr/>
                          <pre>
                <u>
                  <b>Classification Report</b>
                </u>
                <br />
                <br />
                            {report}<br/><hr/>
              </pre>

                          <img src={graph} />
                          <img src={confusionMatrix} />
                        </div>
                      </div>

                  )}
                </React.Fragment>
            )}

            {/* Conditionally render Semi-Supervised Models */}
            {learning === 'activeLearning' && (
                <>
                  <h3 style={{color:"black"}}>Semi - Supervised Models</h3>
                  <ActiveLearning />
                </>
            )}
          </div>


        </div></>
  );
};

export default MLdropdown;
