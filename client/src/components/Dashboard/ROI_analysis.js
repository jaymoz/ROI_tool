import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart, PointElement, LineElement } from 'chart.js';
import axios from 'axios';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import './ROI_analysis.css';
import ROI_graphs from './ROI_graphs';

Chart.register(PointElement, LineElement);

const ROI = () => {
  const [fpCost, setFpCost] = useState(5);
  const [fnCost, setFnCost] = useState(5);
  const [tpCost, setTpCost] = useState(5);
  const [resourcesCost, setResourcesCost] = useState(5);
  const [preprocessingCost, setPreprocessingCost] = useState(5);
  const [productValue, setProductValue] = useState(5);
  const [showGraphs, setShowGraphs] = useState(false);

 
  const handleApply = () => {
    setShowGraphs(false);
    const formData = new FormData();
    formData.append('fp_cost', fpCost);
    formData.append('fn_cost', fnCost);
    formData.append('tp_cost', tpCost);
    formData.append('resources_cost', resourcesCost);
    formData.append('preprocessing_cost', preprocessingCost);
    formData.append('product_value', productValue);

    axios.post('http://44.201.124.234:5000/roi-parameters', formData)
      .then(response => {
        console.log(response.data);
        setShowGraphs(true);
      })
      .catch(error => {
        console.error(error);
      });
  };

  return (
    <>
    <div className="roi-container">
    <div className="slidercontainer">
    <div className="slider-wrapper-left">
      <div className="row" style={{marginBottom: '10px'}}><span className="slider-value">Fp Cost : {fpCost}</span></div>
      <div className="row">
        <Slider
          min={5}
          max={100}
          value={fpCost}
          onChange={value => setFpCost(parseFloat(value))}
          trackStyle={{ backgroundColor: '#28a9e2', height: '8px' }}
          handleStyle={{
            borderColor: '#28a9e2',
            height: '20px',
            width: '20px',
            marginLeft: '-5px',
            marginTop: '-6px',
          }}
          railStyle={{ backgroundColor: 'lightgray', height: '8px' }}
          className="slider"
        />
      </div>
      <div className="row" style={{marginBottom: '10px'}}><span className="slider-value">Fn Cost : {fnCost}</span></div>
      <div className="row">
        <Slider
          min={5}
          max={100}
          value={fnCost}
          onChange={value => setFnCost(parseFloat(value))}
          trackStyle={{ backgroundColor: '#28a9e2', height: '8px' }}
          handleStyle={{
            borderColor: '#28a9e2',
            height: '20px',
            width: '20px',
            marginLeft: '-5px',
            marginTop: '-6px',
          }}
          railStyle={{ backgroundColor: 'lightgray', height: '8px' }}
          className="slider"
        />
      </div>
      <div className="row" style={{marginBottom: '10px'}}><span className="slider-value">Tp Cost : {tpCost}</span></div>
      <div className="row">
        <Slider
          min={5}
          max={100}
          value={tpCost}
          onChange={value => setTpCost(parseFloat(value))}
          trackStyle={{ backgroundColor: '#28a9e2', height: '8px' }}
          handleStyle={{
            borderColor: '#28a9e2',
            height: '20px',
            width: '20px',
            marginLeft: '-5px',
            marginTop: '-6px',
          }}
          railStyle={{ backgroundColor: 'lightgray', height: '8px' }}
          className="slider"
        />
      </div>
      </div>
      <div className="slider-wrapper-right">
      <div className="row" style={{marginBottom: '10px'}}><span className="slider-value">Resources Cost : {resourcesCost}</span></div>
      <div className="row">
        <Slider
          min={5}
          max={100}
          value={resourcesCost}
          onChange={value => setResourcesCost(parseFloat(value))}
          trackStyle={{ backgroundColor: '#28a9e2', height: '8px' }}
          handleStyle={{
            borderColor: '#28a9e2',
            height: '20px',
            width: '20px',
            marginLeft: '-5px',
            marginTop: '-6px',
          }}
          railStyle={{ backgroundColor: 'lightgray', height: '8px' }}
          className="slider"
        />
      </div>
      <div className="row" style={{marginBottom: '10px'}}><span className="slider-value">Preprocessing Cost : {preprocessingCost}</span></div>
      <div className="row">
        <Slider
          min={5}
          max={100}
          value={preprocessingCost}
          onChange={value => setPreprocessingCost(parseFloat(value))}
          trackStyle={{ backgroundColor: '#28a9e2', height: '8px' }}
          handleStyle={{
            borderColor: '#28a9e2',
            height: '20px',
            width: '20px',
            marginLeft: '-5px',
            marginTop: '-6px',
          }}
          railStyle={{ backgroundColor: 'lightgray', height: '8px' }}
          className="slider"
        />
      </div>
      <div className="row" style={{marginBottom: '10px'}}><span className="slider-value">Product Value : {productValue}</span></div>
      <div className="row">
        <Slider
          min={5}
          max={100}
          value={productValue}
          onChange={value => setProductValue(parseFloat(value))}
          trackStyle={{ backgroundColor: '#28a9e2', height: '8px' }}
          handleStyle={{
            borderColor: '#28a9e2',
            height: '20px',
            width: '20px',
            marginLeft: '-5px',
            marginTop: '-6px',
          }}
          railStyle={{ backgroundColor: 'lightgray', height: '8px' }}
          className="slider"
        />
      </div>
      </div>
      </div>
      <button className="button" onClick={handleApply}>Apply</button>
    
    {showGraphs && <ROI_graphs />}
    </div>
    </>
  );
};

export default ROI;
