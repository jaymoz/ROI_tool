import React, { useState, useEffect } from 'react';
import ActiveLearning from './ActiveLearning';
import axios from 'axios';
import Select from 'react-select';
import './MLConfig.css';
import './Dashboard_sidebar.css';
import arrow_key from '../images/left_arrow.png';
import Results from './Results';
import './ImportCSV.css';

const MLdropdown = () => {
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
    const [cv_mean, setCV] = useState(null);
    const [baseModel, setBaseModel] = useState(null); 


    const [supervisedSubOption, setSupervisedSubOption] = useState(null);
    const baseModelOptions = [
        { value: 'supervised', label: 'Supervised Learning' },
        { value: 'activeLearning', label: 'Active Learning' }
    ];

    const supervisedSubOptions = [
        { value: 'logistic_regression', label: 'logistic_regression' },
        { value: 'naive_bayes', label: 'naive_bayes' },
        { value: 'random_forest', label: 'random_forest' },
        { value: 'support_vector_machine', label: 'support_vector_machine' },
        { value: 'decision_tree', label: 'decision_tree' }
    ];


    const handleModelChange = (selectedOption) => {
        setBaseModel(selectedOption.value);
        console.log(baseModel);
        handleLearningSelect(selectedOption.value);
    };

    const handleSupervisedSubModelChange = (selectedOption) => {
        setSupervisedSubOption(selectedOption.value);
        console.log(supervisedSubOption);
        handleOptionSelect(supervisedSubOption);
    };

  const handleLearningSelect = async (selectedValue) => {
    console.log('Selected option:', selectedValue);

    const headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
        "X-Requested-With": "XMLHttpRequest"  // Required by some versions of cors-anywhere
    };

    if (selectedValue === 'supervised') {
      try {
        const response = await axios.post('https://roibackend.shaktilab.org/weekly-supervised', {}, { headers: headers });
        console.log(response.data);

        if (response.data.success) {
          setLearning('supervised');
        }
      } catch (error) {
        console.error(error);
      }
    } else if (selectedValue === 'activeLearning') {
      try {
        const response = await axios.post('https://roibackend.shaktilab.org/active-learning', {}, { headers: headers });
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
      const response = await axios.post(`https://roibackend.shaktilab.org/${endpoint}`, {}, { headers: headers });
      
      console.log(`https://roibackend.shaktilab.org/${endpoint}`);
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
          setCV(response.data.cv_mean);
      }
  } catch (error) {
      console.error(error);
  }
};

  return (
        <div className="container ">
            <div className='section'>
            <div className='upload-sidebar subsection'>
                <Select
                    options={baseModelOptions}
                    value={baseModel}
                    onChange={handleModelChange}
                    placeholder="ML Type"
                    className='ml-sidebar-item'
                />
                {
                    baseModel === 'supervised' && 
                    <Select
                        options={supervisedSubOptions}
                        value={supervisedSubOption}
                        onChange={handleSupervisedSubModelChange}
                        placeholder="Sub Model"
                        className='ml-sidebar-item'
                    />
                }
                {
                    baseModel === 'activeLearning' && 
                        <ActiveLearning />
                }

            </div>
            </div>
            
            <div className="ml-table-section section" style={{color: '#28a9e2'}}>
                {learning === 'supervised' && (
                    /* supervised models and report */
                    <React.Fragment>
                    {report && (
                        <div className="subsection report-subsection">
                            <div className="report-content">
                                <div className='accuracy-report-subsection report-inside-item'>
                                    <div className="report-tile">
                                        <div className="report-tile-title">Accuracy</div>
                                        <div className="report-tile-value">{accuracy}</div>
                                    </div>
                                    <div className="report-tile">
                                        <div className="report-tile-title">CV score</div>
                                        <div className="report-tile-value">{cv_mean}</div>
                                    </div>
                                    <div className="report-tile">
                                        <div className="report-tile-title">F1 Score</div>
                                        <div className="report-tile-value">{f1_score}</div>
                                    </div>
                                    <div className="report-tile">
                                        <div className="report-tile-title">Execution Time</div>
                                        <div className="report-tile-value">{stopTime} seconds</div>
                                    </div>
                                </div>
                                
                                <pre className='classification-table report-inside-item'>
                                    <u>
                                    <b className='report-tile-title'>Classification Report</b> <br /><br />
                                    </u>
                                    <div className='inner-classification-table'>
                                        {report}
                                    </div>
                                </pre>
                                

                            <img src={graph} />
                            <img src={confusionMatrix}/>
                            </div>
                        </div>
                    )}
                    </React.Fragment>
                )}
                <Results/>
            </div>
        </div>
  );
};

export default MLdropdown;