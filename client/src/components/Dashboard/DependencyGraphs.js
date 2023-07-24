import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {Chart, PointElement, LineElement} from 'chart.js';

Chart.register(PointElement, LineElement);

const DependencyGraphs = () => {
    
  return (
    <p>DependencyGrpah</p>
  );
};

export default DependencyGraphs;
