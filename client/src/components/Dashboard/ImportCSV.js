import React, { useState,useEffect } from 'react';
import axios from 'axios';
import './ImportCSV.css';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import Charts from './ChartComponent';
import Filter from './ColumnFilter';
import arrow_key from '../images/left_arrow.png';

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
  const [sideBarOpen, setSideBar] = useState(false);

  const handleSideBarButton = () =>{
      if (sideBarOpen){
          setSideBar(false);
      }
      else{
          setSideBar(true);
      }
  }

  // useEffect(()=>{
  //     let sidebar_elem = document.querySelector('.upload-sidebar');
  //     let sidebar_img = sidebar_elem.lastElementChild;
  //     let rect = sidebar_elem.getBoundingClientRect();

  //     if (sideBarOpen){
  //         sidebar_elem.style.transform= `translateX(${-rect.x+20}px)`; 
  //         sidebar_img.style.transform = 'rotate(0deg)';
  //     }
  //     else{
  //         sidebar_elem.style.transform= `translateX(0%)`;
  //         sidebar_img.style.transform = 'rotate(180deg)';
  //     }
  // },[sideBarOpen]);

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
        .post('https://roibackend.shaktilab.org/upload/train_data', formData, {
            headers: {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
                "X-Requested-With": "XMLHttpRequest"  // Required by some versions of cors-anywhere
            }
        })
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
      .post('https://roibackend.shaktilab.org/upload/test_data', formData, {
          headers: {
              "Access-Control-Allow-Origin": "*",
              "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
              "X-Requested-With": "XMLHttpRequest"  // Required by some versions of cors-anywhere
          }
      })
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
      .post('https://roibackend.shaktilab.org/trim_data', { trainingSize }, {
          headers: {
              "Access-Control-Allow-Origin": "*",
              "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
              "X-Requested-With": "XMLHttpRequest"  // Required by some versions of cors-anywhere
          }
      })
      .then((response) => {
          console.log(response.data);
          window.alert('Trimmed Data Saved at Backend!');
      })
      .catch((error) => {
          console.error(error);
      });
};


  return (
      <div className="container">
        <div className="section">
          <div className='upload-sidebar subsection'>
              <h2 className="medium-text">Upload Data</h2>
            <div className="slider-container">
              <p className="small-text" style={{ fontFamily: 'inherit' }}>Subset Size: {trainingSize}</p>
              <Slider min={0.1} max={1} step={0.1} value={trainingSize} onChange={handleTrainingSizeChange} />

            </div>
            {trainDataUploaded && (
                <div className="result">
                  <p className="small-text">Data Rows: {train_data_RowCount}</p>
                  <p className="small-text">Data Columns: {train_data_ColCount}</p>
                </div>
            )}
            <input type="file" id="myfile" onChange={handleTrainDataUpload}/>
            <button className="small-button" onClick={handleGraphUpload}>Upload</button>
          </div>
          {showColumnSelection && (
            <div className="box table-section subsection">
              <h2 className="medium-text">Select the required columns</h2>
              <Filter trainData={trainData} />
              <center>
                <button className="button" onClick={handlePreprocessData} style={{width: '250px',height: '50px', marginLeft: '30px'}}>
                  Trim & Preprocess Data
                </button>
              </center>
            </div>

        )}
        </div>
        

        {showColumnSelection && (
            <div className="section">
                {trainDataUploaded && (
                      <Charts />
                )}
              </div>
        )}        
      </div>

  );
}

export default ImportCSV;
