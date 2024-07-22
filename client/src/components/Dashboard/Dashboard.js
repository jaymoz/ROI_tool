import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import ImportCSV from './ImportCSV';
// import MLdropdown from './MLdropdown';
// import Results from './Results';
// import DependencyGraphs from './DependencyGraphs';
// import ROI from './ROI_analysis';
import MLConfig from './MLConfig';
import ROI from './ROI_analysis';



import './Dashboard.css';


function Dashboard(){
   
    return(<div className='Dashboard'>
        <Routes >
            <Route path="/" element={<ImportCSV />} />
            <Route path="/ml-analysis" element={<MLConfig />} />
            <Route path="/roi-analysis" element={<ROI />} />
        </Routes>
        
    </div>);

}

export default Dashboard;