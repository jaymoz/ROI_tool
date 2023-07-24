import React, { useState } from 'react';
import axios from 'axios';
import './ImportCSV.css';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import Charts from './ChartComponent';
import Filter from './ColumnFilter';

function ImportCSV() {
  const [trainData, setTrainData] = useState(null);
  const [trainingSize, setTrainingSize] = useState(0.8);
  const [train_data_RowCount, settrain_data_RowCount] = useState(null);
  const [train_data_ColCount, settrain_data_ColCount] = useState(null);
  const [trainDataUploaded, setTrainDataUploaded] = useState(false);
  const [showColumnSelection, setShowColumnSelection] = useState(false);

  const [testData, setTestData] = useState(null);
  const [test_data_RowCount, settest_data_RowCount] = useState(null);
  const [test_data_ColCount, settest_data_ColCount] = useState(null);
  const [testDataUploaded, setTestDataUploaded] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authenticated, setAuthenticated] = useState(false);

  const handleTrainDataUpload = (event) => {
    setTrainData(event.target.files[0]);
  };

  const handleTrainingSizeChange = (value) => {
    setTrainingSize(value);
  };

  const handleTestDataUpload = (event) => {
    setTestData(event.target.files[0]);
  };

  const handleGraphUpload = () => {
    const formData = new FormData();
    formData.append('file', trainData);
    formData.append('training_size', trainingSize);

    axios
      .post('http://44.201.124.234:5000/upload/train_data', formData)
      .then((response) => {
        console.log(response.data);
        if (response.data.success) {
          settrain_data_RowCount(response.data.rows);
          settrain_data_ColCount(response.data.columns);
          setTrainData(response.data.csv_data);
          setTrainDataUploaded(true);
          setShowColumnSelection(true);
        }
      })
      .catch((error) => {
        console.error(error);
      });
  };

  const handleTestData = () => {
    const formData = new FormData();
    formData.append('file', testData);

    axios
      .post('http://44.201.124.234:5000/upload/test_data', formData)
      .then((response) => {
        console.log(response.data);
        if (response.data.success) {
          settest_data_RowCount(response.data.rows);
          settest_data_ColCount(response.data.columns);
          setTestDataUploaded(true);
        }
      })
      .catch((error) => {
        console.error(error);
      });
  };

  const handlePreprocessData = () => {
    axios
      .post('http://44.201.124.234:5000/trim_data', { trainingSize })
      .then((response) => {
        console.log(response.data);
        window.alert('Trimmed Data Saved at Backend!');
      })
      .catch((error) => {
        console.error(error);
      });
  };
  const handleCredentialsChange = (event) => {
    const { name, value } = event.target;
    if (name === 'username') {
      setUsername(value);
    } else if (name === 'password') {
      setPassword(value);
    }
  };

  const handleLogin = () => {
    // Perform validation against the provided credentials
    if (username === 'gouri' && password === 'gouri') {
      setAuthenticated(true);
    } else {
      setAuthenticated(false);
    }
  };


  return (
    <div className="container">
  <div className="section">
    <div className="box">
      <h2 className="medium-text">Train Data</h2>
      <div className="slider-container">
        <p className="small-text" style={{ fontFamily: 'inherit' }}>Training Size: {trainingSize}</p>
        <Slider min={0.1} max={1} step={0.1} value={trainingSize} onChange={handleTrainingSizeChange} />
      </div>
      <div className="input-section">
        <input type="file" onChange={handleTrainDataUpload} />
        <button className="small-button" onClick={handleGraphUpload}>Upload</button>
      </div>
      {trainDataUploaded && (
        <div className="result">
          <p className="small-text">Data Rows: {train_data_RowCount}</p>
          <p className="small-text">Data Columns: {train_data_ColCount}</p>
        </div>
      )}
    </div>
    <div className="box">
      <h2 className="medium-text">Test Data</h2>
      <div className="input-section">
        <input type="file" onChange={handleTestDataUpload} />
        <button className="small-button" onClick={handleTestData}>Upload</button>
      </div>
      {test_data_RowCount !== null && test_data_ColCount !== null && (
        <div className="result">
          <p className="small-text">Data Rows: {test_data_RowCount}</p>
          <p className="small-text">Data Columns: {test_data_ColCount}</p>
        </div>
      )}
    </div>
  </div>

  {showColumnSelection && (
    <div className="section">
      <div className="box">
        <h2 className="medium-text">Select the required columns</h2>
        <Filter trainData={trainData} />
        <br />
        <br />
        <center>
          <button className="button" onClick={handlePreprocessData} style={{width: '250px',height: '50px', marginLeft: '30px'}}>
            Trim & Preprocess Data
          </button>
        </center>
      </div>
    </div>
  )}

  <div className="graphs-container">
    {trainDataUploaded && (
      <div className="graph-section">
        <Charts />

      </div>
    )}
  </div>
</div>

  );
}

export default ImportCSV;
