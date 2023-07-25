import React, { useState } from 'react';
import Dropdown from './GridDropdown';
import LearningDropdown from './Learning';
import ActiveLearning from './ActiveLearning';
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
    { label: 'Weekly Supervised Model', value: 'weeklySupervised' },
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

    if (selectedValue === 'weeklySupervised') {
      setLearning(selectedValue)
      // try {
      //   const response = await axios.post('http://127.0.0.1:5000/weekly-supervised');
      //   console.log(response.data);
      //   if (response.data.success) {
      //     setLearning(response.data.testCheck);
      //   }
      // } catch (error) {
      //   console.error(error);
      // }
    } else if (selectedValue === 'activeLearning') {
          setLearning(selectedValue)
      //try {
      //   const response = await axios.post('http://127.0.0.1:5000/active-learning');
      //   console.log(response.data);
      //   if (response.data.success) {
      //     setLearning(response.data.testCheck);
      //   }
      // } catch (error) {
      //   console.error(error);
      // }
    }
  };

  const handleOptionSelect = async (selectedValue) => {
    console.log('Selected option:', selectedValue);
    onModelSelect(selectedValue);

    if (selectedValue === 'logistic_regression') {
      try {
        const response = await axios.post('http://44.201.124.234:5000/logistic-regression');
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
    } else if (selectedValue === 'naive_bayes') {
      try {
        const response = await axios.post('http://44.201.124.234:5000/naive-bayes');
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
    } else if (selectedValue === 'random_forest') {
      try {
        const response = await axios.post('http://44.201.124.234:5000/random-forest');
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
    } else if (selectedValue === 'support_vector_machine') {
      try {
        const response = await axios.post('http://44.201.124.234:5000/support-vector-machine');
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
    } else if (selectedValue === 'decision_tree') {
      try {
        const response = await axios.post('http://44.201.124.234:5000/decision-tree');
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
    }
  };

  return (
    <div className="ml-dropdown-container">
      <div className="ml-models">
        <br />
        <div className="text" style={{ fontSize: '38px', textAlign: 'center' }}>
          ML Models
        </div>
        <br />

        {/* learning */}
        <LearningDropdown options={learningAlgo} onSelect={handleLearningSelect} />
        {/* Conditionally render Supervised Models */}
        {learning === 'weeklySupervised' && (
          <>
            <h3 style={{color:"black"}}>Supervised Models</h3>
            <Dropdown options={options} onSelect={handleOptionSelect} />

            {report && (
              <div className="classification-report">
                <div className="report-content">
                  <pre><u><b>Training Accuracy</b></u>: {accuracy}</pre>
                  <pre><u><b>Execution Time</b></u>: {stopTime} seconds</pre>
                  <pre><u><b>F1 Score</b></u>: {f1_score} </pre>
                  <pre><u><b>Classification Report</b></u><br></br><br></br>{report}</pre>
                  <img src={graph}/>
                  <img src={confusionMatrix}/>
                </div>
              </div>
            )}
          </>
        )}

        {/* Conditionally render Semi-Supervised Models */}
        {learning === 'activeLearning' && (
          <>
            <h3 style={{color:"black"}}>Semi - Supervised Models</h3>
            <ActiveLearning />
          </>
        )}
      </div>
    </div>
);

  
  
};

export default MLdropdown;