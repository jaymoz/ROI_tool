import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart, PointElement, LineElement } from 'chart.js';
import axios from 'axios';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import './ROI_analysis.css';
import ROI_graphs from './ROI_graphs';
import { Tooltip } from 'react-tooltip';

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
    
        axios.post('https://roibackend.shaktilab.org/roi-parameters', formData, {
            headers: {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
                "X-Requested-With": "XMLHttpRequest" // This header is sometimes required by cors-anywhere
            }
        })
        .then(response => {
            console.log(response.data);
            setShowGraphs(true);
        })
        .catch(error => {
            console.error(error);
        });
    };
    
    return (
        <div className="roi-container">
            <div className="slidercontainer">
                <div className="slider-wrapper-left">
                    <div className="row" style={{marginBottom: '10px'}}><span data-tooltip-id="tooltip-fpCost" className="slider-value">Fixed Cost : {fpCost}</span>
    <Tooltip id="tooltip-fpCost" place="top" effect="solid">Cost consisting of Data Gathering cost, Pre-processing Cost & Evaluation Cost</Tooltip>
    </div>
                    <div className="row">
                        <Slider
                            min={0.1}
                            max={2}
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
                    <div className="row" style={{ marginBottom: '10px' }}>
            <span data-tooltip-id="tooltip-fnCost" className="slider-value">B_penalty : {fnCost}</span>
            <Tooltip id="tooltip-fnCost" place="top" effect="solid">Penalty per False Negative instance </Tooltip>
          </div>
                    <div className="row">
                        <Slider
                            min={100}
                            max={600}
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
                    <div className="row" style={{ marginBottom: '10px' }}>
            <span data-tooltip-id="tooltip-tpCost" className="slider-value">B_reward : {tpCost}</span>
            <Tooltip id="tooltip-tpCost" place="top" effect="solid">Reward per True Positive instance</Tooltip>
          </div>
                    <div className="row">
                        <Slider
                            min={100}
                            max={600}
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
                    <div className="row" style={{ marginBottom: '10px' }}>
            <span data-tooltip-id="tooltip-resourcesCost" className="slider-value">Resources Cost : {resourcesCost}</span>
            <Tooltip id="tooltip-resourcesCost" place="top" effect="solid">Resources Cost</Tooltip>
          </div>
                    <div className="row">
                        <Slider
                            min={100}
                            max={600}
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
                    <div className="row" style={{ marginBottom: '10px' }}>
            <span data-tooltip-id="tooltip-preprocessingCost" className="slider-value">Labelling Cost : {preprocessingCost}</span>
            <Tooltip id="tooltip-preprocessingCost" place="top" effect="solid">Labelling Cost</Tooltip>
          </div>
                    <div className="row">
                        <Slider
                            min={0.1}
                            max={3}
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
                    <div className="row" style={{ marginBottom: '10px' }}>
            <span data-tooltip-id="tooltip-productValue" className="slider-value">Human Resource : {productValue}</span>
            <Tooltip id="tooltip-productValue" place="top" effect="solid">Total Human Resource involved</Tooltip>
          </div>
                    <div className="row">
                        <Slider
                            min={1}
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

    );
};

export default ROI;
